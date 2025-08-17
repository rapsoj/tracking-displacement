import os
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
import h5py
import matplotlib.pyplot as plt

class PairedImageDataset(Dataset):
    def __init__(self, hdf5_path, feat_transform=None, label_transform=None):

        # Default feature transform: ToTensor
        if feat_transform is None:
            feat_transform = transforms.ToTensor()
        # Default label transform: Gaussian blur + ToTensor
        if label_transform is None:
            class LabelBlur:
                def __call__(self, img):
                    return img.filter(ImageFilter.GaussianBlur(radius=3))
            label_transform = transforms.Compose([
                LabelBlur(),
                transforms.ToTensor(),
                lambda x: x * 20
            ])
        self.hdf5_path = hdf5_path
        self.feat_transform = feat_transform
        self.label_transform = label_transform
        self.h5 = h5py.File(self.hdf5_path, 'r')
        self.greyscale_group = self.h5['feature']
        self.label_group = self.h5['label']
        # Only keep pairs that exist in both groups
        self.keys = sorted(set(self.greyscale_group.keys()) & set(self.label_group.keys()))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        # Load greyscale and label arrays
        grey = self.greyscale_group[key][()]
        label = self.label_group[key][()]
        # Convert to PIL Images for compatibility with transforms
        grey_img = Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8))
        label_img = Image.fromarray(label.astype(np.uint8))
        feat = self.feat_transform(grey_img)
        lab = self.label_transform(label_img)
        # Optionally, return metadata as well
        meta = {attr: self.greyscale_group[key].attrs[attr] for attr in self.greyscale_group[key].attrs}
        return {'feature': feat, 'label': lab, 'meta': meta}

    def close(self):
        self.h5.close()

    def show(self, idx: int, overlay: bool = False) -> None:
        if overlay:
            self.show_overlay(idx)
        else:
            self.show_split(idx)

    def show_overlay(self, idx: int) -> None:
        sample = self[idx]
        feature = sample['feature']
        label = sample['label']
        meta = sample['meta']

        if isinstance(feature, torch.Tensor):
            arr_feat = feature.squeeze().cpu().numpy()
        else:
            arr_feat = np.array(feature)
        if isinstance(label, torch.Tensor):
            arr_label = label.squeeze().cpu().numpy()
        else:
            arr_label = np.array(label)
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 6]})
        axes[0].axis('off')
        meta["origin_date"] = f"{meta['origin_date'][:4]}-{meta['origin_date'][4:6]}-{meta['origin_date'][6:]}"
        meta["origin_image"] = "_".join([comp for comp in meta["origin_image"].split("_")[:4] if not comp.isdigit()])
        meta_text = '\n'.join(f"{k}: {v}" for k, v in meta.items())
        axes[0].text(0.1, 0.9, meta_text, fontsize=12, color='black',
                    ha='left', va='top', transform=axes[0].transAxes)
        axes[1].imshow(arr_feat, cmap='gray', interpolation='none')
        axes[1].set_title('Data')
        axes[1].axis('off')

        plt.imshow(np.ones_like(arr_label), cmap="spring", alpha=arr_label / np.max(arr_label), interpolation='none')
        plt.tight_layout()
        plt.show()

    def show_split(self, idx: int) -> None:
        sample = self[idx]
        feature = sample['feature']
        label = sample['label']
        meta = sample['meta']
        # Convert tensors to numpy for plotting
        if isinstance(feature, torch.Tensor):
            arr_feat = feature.squeeze().cpu().numpy()
        else:
            arr_feat = np.array(feature)
        if isinstance(label, torch.Tensor):
            arr_label = label.squeeze().cpu().numpy()
        else:
            arr_label = np.array(label)
        fig, axes = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 3, 3]})
        # Left: meta text
        axes[0].axis('off')
        meta["origin_date"] = f"{meta['origin_date'][:4]}-{meta['origin_date'][4:6]}-{meta['origin_date'][6:]}"
        meta["origin_image"] = "_".join([comp for comp in meta["origin_image"].split("_")[:4] if not comp.isdigit()])
        meta_text = '\n'.join(f"{k}: {v}" for k, v in meta.items())
        axes[0].text(0.1, 0.9, meta_text, fontsize=12, color='black',
                    ha='left', va='top', transform=axes[0].transAxes)
        # Center: feature image
        axes[1].imshow(arr_feat, cmap='gray', interpolation='none')
        axes[1].set_title('Feature')
        axes[1].axis('off')
        # Right: label image
        axes[2].imshow(arr_label[::-1, :], cmap='gray')
        axes[2].set_title('Label')
        axes[2].axis('off')
        plt.tight_layout()
        plt.show()

# Example usage:
# dataset = PairedImageDataset('/path/to/file.hdf5')
# sample = dataset[0]
# img, label, meta = sample['feature'], sample['label'], sample['meta']


if __name__ == "__main__":
    ds = PairedImageDataset("processed_data.h5")

    print(f"Length: {len(ds)}")
    for i in range(10):
        j = np.random.randint(0, len(ds))
        ds.show(j, overlay=True)
