import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scrape_fa.paired_image_dataset import PairedImageDataset
from scrape_fa.simple_cnn import SimpleCNN

plt.ion()  # interactive mode

def normalize(img):
    """Normalize image to [0,1]"""
    img = img.astype(np.float32)
    img -= img.min()
    img /= (img.max() + 1e-8)
    return img

def overlay(base, mask, alpha=0.5, color='red'):
    """Overlay a mask on the base image with specified color"""
    base_rgb = np.stack([normalize(base)]*3, axis=-1)
    overlay_rgb = np.zeros_like(base_rgb)
    channel = {'red':0, 'green':1, 'blue':2}[color]
    overlay_rgb[..., channel] = normalize(mask)
    return (1-alpha)*base_rgb + alpha*overlay_rgb

def visualize_training_subset(hdf5_path, model_path, sample_size=100, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    ds = PairedImageDataset(hdf5_path)

    # Random subset of indices
    indices = random.sample(range(len(ds)), min(sample_size, len(ds)))

    # Load model
    model = SimpleCNN.from_pth(model_path, model_args={"n_channels": 2, "n_classes": 1})
    model.to(device)
    model.eval()

    for idx in indices:
        sample = ds[idx]

        # Prepare input: concatenate feature + prewar
        feats = torch.cat((sample['feature'], sample['prewar'])).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(feats).squeeze().cpu().numpy()

        # Convert tensors to numpy arrays
        feature = sample['feature'].squeeze().cpu().numpy() if torch.is_tensor(sample['feature']) else sample['feature']
        prewar = sample['prewar'].squeeze().cpu().numpy() if torch.is_tensor(sample['prewar']) else sample['prewar']
        label = sample['label'].squeeze().cpu().numpy() if torch.is_tensor(sample['label']) else sample['label']

        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(normalize(prewar), cmap='gray')
        axes[0].set_title('Prewar')

        axes[1].imshow(normalize(feature), cmap='gray')
        axes[1].set_title('Current')

        axes[2].imshow(overlay(feature, pred, color='red'))
        axes[2].set_title('Prediction Overlay')

        axes[3].imshow(overlay(feature, label, color='green'))
        axes[3].set_title('Label Overlay')

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize model predictions on training data")
    parser.add_argument("--dataset", required=True, help="Path to processed_data.h5")
    parser.add_argument("--model", required=True, help="Path to trained model .pth file")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of random samples to visualize")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize_training_subset(
        hdf5_path=args.dataset,
        model_path=args.model,
        sample_size=args.sample_size,
        device=device
    )


#
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.ion()  # enable interactive mode
#
# # --- Utility functions ---
# def normalize(img):
#     img = img.astype(np.float32)
#     img -= img.min()
#     img /= (img.max() + 1e-8)
#     return img
#
# def overlay(base, mask, alpha=0.5, color='red'):
#     """Overlay mask on base image"""
#     base_uint8 = (normalize(base) * 255).astype(np.uint8)
#     mask_uint8 = (normalize(mask) * 255).astype(np.uint8) if mask.max() > 0 else np.zeros_like(base_uint8, dtype=np.uint8)
#     base_rgb = np.stack([base_uint8]*3, axis=-1)
#     overlay_rgb = np.zeros_like(base_rgb)
#     overlay_rgb[..., {'red':0, 'green':1, 'blue':2}[color]] = mask_uint8
#     return ((1-alpha)*base_rgb + alpha*overlay_rgb).astype(np.uint8)
#
# # --- Tile viewer class ---
# class TileViewer:
#     def __init__(self, processed_h5_path, predictions_h5_path):
#         self.processed_h5 = h5py.File(processed_h5_path, "r")
#         self.predictions_h5 = h5py.File(predictions_h5_path, "r")
#         self.tiles = list(self.processed_h5["feature"].keys())
#         self.index = 0
#
#         self.fig, self.axes = plt.subplots(1, 4, figsize=(20,5))
#         self.fig.canvas.mpl_connect("key_press_event", self.on_key)
#
#         self.update()
#
#     def update(self):
#         tile_name = self.tiles[self.index]
#
#         current = self.processed_h5["feature"][tile_name][()]
#         prewar = self.processed_h5["prewar"][tile_name][()]
#         label = self.processed_h5["label"][tile_name][()]
#         pred = self.predictions_h5["predictions"][tile_name][()]
#
#         if current.ndim > 2: current = current[0]
#         if prewar.ndim > 2: prewar = prewar[0]
#         if label.ndim > 2: label = label[0]
#         if pred.ndim > 2: pred = pred[0]
#
#         overlay_pred = overlay(current, pred, color='red')
#         overlay_label = overlay(current, label, color='green')
#
#         self.axes[0].imshow(prewar, cmap='gray')
#         self.axes[0].set_title("Prewar")
#         self.axes[0].axis('off')
#
#         self.axes[1].imshow(current, cmap='gray')
#         self.axes[1].set_title("Current")
#         self.axes[1].axis('off')
#
#         self.axes[2].imshow(overlay_pred)
#         self.axes[2].set_title("Prediction Overlay")
#         self.axes[2].axis('off')
#
#         self.axes[3].imshow(overlay_label)
#         self.axes[3].set_title("Label Overlay")
#         self.axes[3].axis('off')
#
#         self.fig.suptitle(f"Tile {self.index+1}/{len(self.tiles)}: {tile_name}")
#         self.fig.canvas.draw_idle()
#
#     def on_key(self, event):
#         if event.key == 'right':
#             self.index = (self.index + 1) % len(self.tiles)
#             self.update()
#         elif event.key == 'left':
#             self.index = (self.index - 1) % len(self.tiles)
#             self.update()
#
# # --- Run viewer ---
# processed_path = "processed_data.h5"
# predictions_path = "predictions.h5"
# viewer = TileViewer(processed_path, predictions_path)
# plt.show(block=True)
