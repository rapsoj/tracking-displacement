from __future__ import annotations
from typing import Callable
import random
from pathlib import Path

from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import label

from scrape_fa.util.logging_config import setup_logging

LOGGER = setup_logging("paired-image-ds")


def remove_small_components(arr, min_area):
    # Label connected components
    labeled, num_features = label(arr)
    # Count area of each component (component 0 is background)
    areas = np.bincount(labeled.ravel())
    # Create mask for components above threshold
    mask = np.isin(labeled, np.where(areas >= min_area)[0])
    # Return filtered array (same dtype as input)
    return arr * mask

class PairedImageDataset(Dataset):
    def __init__(self, hdf5_path, feat_transform: Callable | None = None, label_transform: Callable | None = None, indices: list[int] | None = None, is_pred: bool = False):

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
        self.h5 = h5py.File(self.hdf5_path, 'r')  # r+ supports saving of splits
        self.greyscale_group = self.h5['feature']
        self.label_group = self.h5['label']
        self.is_pred = self.h5.attrs.get('is_pred', is_pred)
        self.indices: list | None = indices
        # Only keep pairs that exist in both groups
        self.keys = sorted(set(self.greyscale_group.keys()) & set(self.label_group.keys()))

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.keys)

    def __getitem__(self, idx, remap_idx: bool = True):
        if self.indices is not None and remap_idx:
            old_idx = idx
            idx = self.indices[idx]
        key = self.keys[idx]
        # Load greyscale and label arrays
        grey = self.greyscale_group[key][()].squeeze().astype(np.float64)
        label = self.label_group[key][()].astype(np.float64)


        meta = {attr: self.greyscale_group[key].attrs[attr] for attr in self.greyscale_group[key].attrs}

        if self.is_pred:
            return {'feature': grey, 'label': label, 'meta': meta}
        # Convert to PIL Images for compatibility with transforms
        grey_img = Image.fromarray(grey.astype(np.uint8))
        label_img = Image.fromarray(label.astype(np.uint8))
        feat = self.feat_transform(grey_img)
        lab = self.label_transform(label_img)
        return {'feature': feat, 'label': lab, 'meta': meta}

    def create_subsets(self, splits: list[float], shuffle: bool = True, save_loc: str | None = None, regenerate_splits: bool = False) -> list["PairedImageDataset"]:
        # Todo: create a copy of this dataset in run folder, so we can keep track of cached splits

        if save_loc is None:
            cache_valid = False
        else:
            split_file = Path(save_loc) / "splits.csv"
            cache_valid = split_file.exists() and not regenerate_splits


        idcs_list = []
        if cache_valid:
            LOGGER.info("Found cached splits, using those.")
            with split_file.open("r") as f:
                data = f.readlines()

            fracs = data[0].strip().split(",")
            fracs = [float(field.strip()) for field in fracs]
            if np.allclose(fracs, splits):
                for row in data[1:]:
                    idcs = row.strip().split(",")
                    idcs_list.append([int(field.strip()) for field in idcs])
            else:
                LOGGER.warn("Cached splits don't match args, ignoring cache")


        idcs = list(range(len(self)))
        if shuffle:
            random.shuffle(idcs)

        start_idx = 0
        datasets = []
        for i, split in enumerate(splits):
            end_idx = start_idx + int(len(self) * split)
            subset_indices = idcs[start_idx:end_idx]
            if not cache_valid: # This means we didnt cache them
                idcs_list.append(subset_indices)
            else:
                subset_indices = idcs_list[i]
            datasets.append(
                PairedImageDataset(
                    self.hdf5_path,
                    feat_transform=self.feat_transform,
                    label_transform=self.label_transform,
                    indices=subset_indices
                )
            )

            start_idx = end_idx

        return datasets, idcs_list


    def close(self):
        self.h5.close()

    def show(self, idx: int, overlay: bool = True) -> None:
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

        arr_label = np.where(arr_label > 1, 1., 0.)
        arr_label = remove_small_components(arr_label, 25)

        axes[1].imshow(np.ones_like(arr_label), cmap="spring", alpha=arr_label, interpolation='none')
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

    @classmethod
    def from_predictions(cls, base_ds: "PairedImageDataset", predictions: list[torch.Tensor], output_path: str, post_processor: Callable = lambda x: x):
        """
        Create a new dataset from predictions, saving them to an HDF5 file.
        """
        with h5py.File(output_path, 'w') as h5f:
            feat_group = h5f.create_group('feature')
            label_group = h5f.create_group('label')
            h5f.attrs['is_pred'] = True
            for i, (sample, pred) in enumerate(zip(base_ds, predictions)):
                key = f"sample_{i}"
                feat_group.create_dataset(key, data=sample['feature'].cpu().numpy().squeeze())
                label_group.create_dataset(key, data=post_processor(pred.cpu().numpy()))
                for attr, value in sample['meta'].items():
                    feat_group[key].attrs[attr] = value
                    label_group[key].attrs[attr] = value

        return cls(output_path, label_transform=lambda x: x, feat_transform=base_ds.feat_transform, indices=base_ds.indices, is_pred=True)


if __name__ == "__main__":
    ds = PairedImageDataset("predictions.h5")

    for i in range(len(ds)):
        try:
            ds.show(i)
        except Exception as exc:
            print("Failed to show index", i, ":", exc)
