from __future__ import annotations
from pathlib import Path
import shutil

import click
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import os
import yaml
import datetime
from scrape_fa.paired_image_dataset import PairedImageDataset
from scrape_fa.simple_cnn import SimpleCNN
from scrape_fa.util.logging_config import setup_logging
from scrape_fa.predict import prediction

LOGGER = setup_logging("train-cnn")

def custom_collate(batch):
    # Initialize an empty dictionary to store concatenated results
    collated_dict = {}

    # Iterate over keys in the first dictionary of the batch
    for key in batch[0].keys():
        entry = [d[key] for d in batch]

        if key != "meta":
            entry = torch.stack(entry, dim=0)

        collated_dict[key] = entry

    return collated_dict

@click.command()
@click.argument('config', type=click.Path(exists=True))
def cli(config: str) -> None:
    with open(config, 'r') as f:
        params = yaml.safe_load(f)
    required = ['hdf5', 'training']
    for k in required:
        if k not in params:
            raise click.ClickException(f"Missing required config key: {k}")
    train(
        params['hdf5'],
        **params['training']
    )

def train(hdf5_path: str, training_frac: float, validation_frac: float, batch_size: int, epochs: int, learning_rate: float, checkpoint: str | None = None, device: str | None = None) -> None:

    # Set device to GPU if available
    if torch.cuda.is_available() and device is None or device == "cuda":
        device = torch.device('cuda')
        LOGGER.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
    elif device is None or device == "cpu":
        device = torch.device('cpu')
        LOGGER.info('Using CPU')
    else:
        raise Exception(f"Could not find device {device}")

    if checkpoint:
        checkpoint = Path(checkpoint)
        model = SimpleCNN.from_pth(checkpoint, model_args={"n_channels": 2, "n_classes": 1}).to(device)
        hdf5_path_obj = Path(hdf5_path)

        save_loc = checkpoint.parent
    else:
        model = SimpleCNN(2, 1).to(device)
        save_loc = None

    # Load and shuffle dataset
    dataset = PairedImageDataset(hdf5_path)
    splits = [training_frac, validation_frac, 1-training_frac-validation_frac]
    (train_set, val_set, test_set), idcs_list = dataset.create_subsets(splits, shuffle=True, save_loc=save_loc)
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=custom_collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=custom_collate)

    LOGGER.info(f"Split {len(dataset)} samples into {len(train_set)} train, {len(val_set)} validation, and {len(test_set)} test samples.")

    criterion = lambda x, y: ((x - y).abs() * (1 + y)**2).mean() + torch.relu(-x).mean() * 10.0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create timestamped run directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    # shutil.copy(hdf5_path, os.path.join(run_dir, 'dataset.h5'))

    # caching splits for future use
    with open(os.path.join(run_dir, "splits.csv"), "w") as split_file:
        split_file.write(",".join([str(split) for split in splits]) + "\n")
        for idcs in idcs_list:
            split_file.write(",".join([str(idx) for idx in idcs]) + "\n")

    best_eval = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, entry in enumerate(train_loader):
            feats = torch.cat((entry["feature"], entry["prewar"]), axis=1).to(device)
            labels = entry["label"].to(device)

            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch}: Completed step {i+1} / {len(train_loader)}", end="\r")
        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, entry in enumerate(val_loader):
                feats = torch.cat((entry["feature"], entry["prewar"]), axis=1).to(device)
                labels = entry["label"].to(device)
                outputs = model(feats)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        if val_loss < best_eval:
            model_path = os.path.join(run_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            LOGGER.info(f"Best model saved to {model_path} at epoch {epoch} with loss {val_loss} < {best_eval}.")
            best_eval = val_loss
        LOGGER.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f} - Validation Loss: {val_loss:.4f}")

    # Create timestamped run directory
    # Save trained model
    # Run through test set and save overlays
    model.eval()
    prediction(test_set, model, os.path.join(run_dir, "test_predictions.h5"), device)
    LOGGER.info("Test overlays and comparison figures saved.")

if __name__ == '__main__':
    train()
