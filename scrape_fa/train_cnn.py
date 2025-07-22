import click
import torch
from torch.utils.data import DataLoader, Subset
import random
import torch.nn as nn
import torch.optim as optim
import os
from scrape_fa.show_overlay import overlay_images
import matplotlib.pyplot as plt
import datetime
from scrape_fa.paired_image_dataset import PairedImageDataset
from scrape_fa.simple_cnn import SimpleCNN


def save_comparison_figure(feat_img, label_img, pred_img, save_path):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    plt.sca(axs[0])
    im_1 = plt.imshow(feat_img.squeeze(), cmap='gray')
    plt.colorbar(im_1, ax=axs[0])
    plt.sca(axs[1])
    im_2 = plt.imshow(label_img.squeeze(), cmap='gray')
    plt.colorbar(im_2, ax=axs[1])
    plt.sca(axs[2])
    im_3 = plt.imshow(pred_img.squeeze(), cmap='gray')
    plt.colorbar(im_3, ax=axs[2])
    plt.savefig(save_path)
    plt.close(fig)

@click.command()
@click.argument('folder', type=click.Path(exists=True))
@click.option('--train_frac', default=0.7, type=float, help='Fraction of data for training')
@click.option('--val_frac', default=0.15, type=float, help='Fraction of data for validation')
@click.option('--batch_size', default=4, type=int, help='Batch size for training')
@click.option('--epochs', default=5, type=int, help='Number of epochs')
def main(folder, train_frac, val_frac, batch_size, epochs):
    # Load and shuffle dataset
    dataset = PairedImageDataset(folder)
    print(f"Dataset size: {len(dataset)}")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n = len(indices)
    n_train = int(n * train_frac)
    print(f"Training set size: {n_train}")
    n_val = int(n * val_frac)
    print(f"Validation set size: {n_val}")
    n_test = n - n_train - n_val
    print(f"Test set size: {n_test}")
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    test_names = [dataset.pairs[i][0] for i in test_idx]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Set device to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print('Using CPU')

    model = UNet(1, 1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create timestamped run directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', timestamp)
    os.makedirs(run_dir, exist_ok=True)

    val_img_int = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)

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
            negative_loss = torch.relu(-outputs).mean() * 10.0  # scale penalty
            loss = criterion(outputs, labels) + negative_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, (feats, labels) in enumerate(val_loader):
                feats, labels = feats.to(device), labels.to(device)
                outputs = model(feats)
                if epoch % val_img_int == 0 and idx < 5:
                    comparison_path = os.path.join(run_dir, f"validation_{idx * batch_size}.png")
                    save_comparison_figure(feats[0].cpu(), labels[0].cpu(), outputs[0].cpu(), comparison_path)
                loss = criterion(outputs, labels)
                # Additional loss for negative values in outputs
                negative_loss = torch.relu(-outputs).mean() * 10.0  # scale penalty
                loss = loss + negative_loss
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f} - Validation Loss: {val_loss:.4f}")

    # Create timestamped run directory
    # Save trained model
    model_path = os.path.join(run_dir, 'trained_cnn.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # Run through test set and save overlays
    dataset_folder = run_dir
    model.eval()
    with torch.no_grad():
        for idx, (feats, labels) in enumerate(test_loader):
            feats = feats.to(device)
            outputs = model(feats)
            # outputs: (B, 1, H, W)
            for b in range(feats.shape[0]):
                feat_img = feats[b].cpu()
                pred_img = outputs[b].cpu()
                comparison_path = os.path.join(run_dir, f"testing_{idx * test_loader.batch_size + b}.png")
                save_comparison_figure(feat_img, labels[b].cpu(), pred_img, comparison_path)
                overlay = overlay_images(feat_img, pred_img)
                out_name = f"{test_names[idx * test_loader.batch_size + b]}_pred.png"
                out_path = os.path.join(run_dir, out_name)
                plt.imsave(out_path, overlay)
    print("Test overlays and comparison figures saved.")

if __name__ == '__main__':
    main()
