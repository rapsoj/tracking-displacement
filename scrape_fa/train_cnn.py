import click
import torch
from torch.utils.data import DataLoader, Subset
import random
import torch.optim as optim
import os
import yaml
import datetime
from scrape_fa.paired_image_dataset import PairedImageDataset
from scrape_fa.simple_cnn import SimpleCNN
from scrape_fa.util.logging_config import setup_logging

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

def load_and_split_ds(ds_folder: str, train_frac: float, val_frac: float, batch_size: int, shuffle: bool = True) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = PairedImageDataset(ds_folder)
    indices = list(range(len(dataset)))

    if shuffle:
        random.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_frac)
    LOGGER.info(f"Training set size: {n_train}")
    n_val = int(n * val_frac)
    LOGGER.info(f"Validation set size: {n_val}")
    n_test = n - n_train - n_val
    LOGGER.info(f"Test set size: {n_test}")
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=custom_collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=custom_collate)

    return train_loader, val_loader, test_loader

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

def train(folder: str, training_frac: float, validation_frac: float, batch_size: int, epochs: int, learning_rate: float) -> None:
    # Load and shuffle dataset
    train_loader, val_loader, test_loader = load_and_split_ds(folder, training_frac, validation_frac, batch_size, shuffle=True)

    # Set device to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        LOGGER.info('Using GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        LOGGER.info('Using CPU')

    model = SimpleCNN(1, 1).to(device)
    criterion = lambda x, y: ((x - y).abs() * (1 + y)**2).mean() + torch.relu(-x).mean() * 10.0
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create timestamped run directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', timestamp)
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for entry in train_loader:
            # ToDo: create a custom collate function for this
            feats, labels = entry["feature"].to(device), entry["label"].to(device)

            # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # plt.sca(axs[0])
            # im_1 = plt.imshow(feats[0].cpu().squeeze(), cmap='gray')
            # plt.colorbar(im_1, ax=axs[0])
            # plt.sca(axs[1])
            # im_2 = plt.imshow(labels[0].cpu().squeeze(), cmap='gray')
            # fig.colorbar(im_2, ax=axs[1])
            # plt.show()
            optimizer.zero_grad()
            outputs = model(feats)
            # Predict only the center pixel of each 10x10 window
            # Crop outputs and labels to center 1 pixel
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, entry in enumerate(val_loader):
                feats, labels = entry["feature"].to(device), entry["label"].to(device)
                outputs = model(feats)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        LOGGER.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f} - Validation Loss: {val_loss:.4f}")

    # Create timestamped run directory
    # Save trained model
    model_path = os.path.join(run_dir, 'trained_cnn.pth')
    torch.save(model.state_dict(), model_path)
    LOGGER.info(f"Model saved to {model_path}")
    # Run through test set and save overlays
    dataset_folder = run_dir
    model.eval()
    output_combined = None
    with torch.no_grad():
        for entry in test_loader.dataset:
            feats = feats.to(device)
            outputs = model(feats)
            if output_combined is None:
                output_combined = outputs.cpu()
            else:
                output_combined = torch.cat((output_combined, outputs.cpu()), dim=0)

        PairedImageDataset.from_predictions(test_loader.dataset, output_combined, os.path.join(run_dir, 'test_predictions.h5'))

    LOGGER.info("Test overlays and comparison figures saved.")

if __name__ == '__main__':
    train()
