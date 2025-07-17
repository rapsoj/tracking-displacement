import click
import os
from scrape_fa.paired_image_dataset import PairedImageDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF

def overlay_images(feat_img, label_img):
    # feat_img, label_img: torch tensors, shape (1, H, W), values in [0,1]
    feat_np = feat_img.squeeze().cpu().numpy()
    label_np = label_img.squeeze().cpu().numpy()
    # Convert feature to RGB
    feat_rgb = np.stack([feat_np]*3, axis=-1)
    # Make label green mask
    green_mask = np.zeros_like(feat_rgb)
    green_mask[..., 1] = label_np  # Green channel
    # Overlay: where label is present, show green, else show feature
    overlay = feat_rgb.copy()
    overlay[label_np > 0] = 30 * green_mask[label_np > 0]
    overlay = np.clip(overlay, 0, 1)
    return overlay

@click.command()
@click.argument('folder', type=click.Path(exists=True))
@click.argument('index', type=int)
def show_overlay(folder, index):
    dataset = PairedImageDataset(folder)
    feat_img, label_img = dataset[index]
    # Apply transforms manually to feat and label
    overlay = overlay_images(feat_img, label_img)
    plt.imshow(overlay)
    plt.title(f'Overlay for index {index}')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    show_overlay()
