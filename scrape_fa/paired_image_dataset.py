from __future__ import annotations
from typing import Callable
import random
from pathlib import Path
from functools import lru_cache

from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import label, gaussian_filter
from deprecated import deprecated

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


def create_rebalanced_subset(dataset: "PairedImageDataset", n_frac: float, seed: int | None = None) -> "PairedImageDataset":
    """
    Create a rebalanced dataset with negative labels randomly sampled to constitute a fraction n_frac of the data.

    :param dataset: The PairedImageDataset to rebalance
    :param n_frac: Fraction of data that should be negative samples (0 <= n_frac < 1)
    :param seed: Random seed for sampling
    :return: A new PairedImageDataset with the rebalanced subset
    """
    if not (0.0 <= n_frac < 1.0):
        raise ValueError(f"n_frac must be between 0 and 1 (exclusive of 1), got {n_frac}")

    pos_indices = []
    neg_indices = []

    LOGGER.info("Scanning dataset for positive/negative balance...")
    for i in range(len(dataset)):
        if dataset.label_is_negative(i):
            neg_indices.append(i)
        else:
            pos_indices.append(i)

    n_pos = len(pos_indices)
    n_neg_total = len(neg_indices)
    LOGGER.info(f"Found {n_pos} positive and {n_neg_total} negative samples.")

    if n_pos == 0:
        raise ValueError("Cannot create rebalanced dataset: No positive samples found.")

    # Calculate needed negatives: n_neg / (n_neg + n_pos) = n_frac
    n_neg_needed = int(round((n_frac * n_pos) / (1.0 - n_frac)))

    if n_neg_needed > n_neg_total:
        max_n_neg = n_neg_total
        max_frac = max_n_neg / (max_n_neg + n_pos)
        raise ValueError(
            f"Cannot create rebalanced dataset with n_frac={n_frac}. "
            f"Need {n_neg_needed} negatives but only have {n_neg_total}. "
            f"Maximum possible n_frac is {max_frac:.4f}"
        )

    if seed is not None:
        random.seed(seed)

    sampled_neg_indices = random.sample(neg_indices, n_neg_needed)
    selected_indices_local = pos_indices + sampled_neg_indices
    random.shuffle(selected_indices_local)

    # Resolve to absolute HDF5 indices
    if dataset.indices is not None:
        final_indices = [dataset.indices[i] for i in selected_indices_local]
    else:
        final_indices = selected_indices_local

    LOGGER.info(f"Created rebalanced dataset with {len(pos_indices)} positives and {len(sampled_neg_indices)} negatives (n_frac={n_frac:.2f})")

    return PairedImageDataset(
        dataset.hdf5_path,
        feat_transform=dataset.feat_transform,
        label_transform=dataset.label_transform,
        indices=final_indices
    )


class PairedImageDataset(Dataset):
    def __init__(self, hdf5_path, feat_transform: Callable | None = None, label_transform: Callable | None = None, indices: list[int] | None = None):

        # Default feature transform: numpy array to tensor
        if feat_transform is None:
            def feat_transform(arr):
                return torch.from_numpy(arr).unsqueeze(0)
        # Default label transform: Gaussian blur + to tensor
        if label_transform is None:
            def label_transform(arr):
                blurred = gaussian_filter(arr, sigma=3)
                return torch.from_numpy(blurred).unsqueeze(0) * 20
        self.hdf5_path = hdf5_path
        self.feat_transform = feat_transform
        self.label_transform = label_transform
        self.h5 = h5py.File(self.hdf5_path, 'r')  # r+ supports saving of splits
        self.greyscale_dataset = self.h5['feature']
        self.prewar_dataset = self.h5['prewar']
        self.label_dataset = self.h5['label']
        self.meta_dataset = self.h5["meta"]
        self.indices: list | None = indices

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return self.label_dataset.shape[0]

    @lru_cache(maxsize=None)
    def __getitem__(self, idx, remap_idx: bool = True):
        if self.indices is not None and remap_idx:
            idx = self.indices[idx]
        # Load greyscale and label arrays
        grey = self.greyscale_dataset[idx].squeeze()
        label_arr = self.label_dataset[idx]
        prewar = self.prewar_dataset[idx].squeeze()

        meta = self.meta_dataset[idx]

        # Apply transforms directly to numpy arrays
        feat = self.feat_transform(grey)
        lab = self.label_transform(label_arr)
        prewar = self.feat_transform(prewar)
        return {'feature': feat, 'label': lab, 'meta': meta, "prewar": prewar}

    def label_is_negative(self, i):
        sample = self[i]
        label = sample['label']
        return (label.max() == 0).item()


    def create_subsets(self, splits: list[float], shuffle: bool = True, save_loc: str | None = None, regenerate_splits: bool = False, seed: int | None = None) -> list["PairedImageDataset"]:
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
            if seed is not None:
                random.seed(seed)
            random.shuffle(idcs)

        start_idx = 0
        datasets = []
        for i, split in enumerate(splits):
            end_idx = start_idx + int(len(self) * split)
            subset_indices = idcs[start_idx:end_idx]
            if not cache_valid: # This means we didn't cache them
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
        # meta["origin_date"] = f"{meta['origin_date'][:4]}-{meta['origin_date'][4:6]}-{meta['origin_date'][6:]}"
        # meta["origin_image"] = "_".join([comp for comp in meta["origin_image"].split("_")[:4] if not comp.isdigit()])
        meta_text = "N/A"  # '\n'.join(f"{k}: {v}" for k, v in meta.items())
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
        # meta["origin_date"] = f"{meta['origin_date'][:4]}-{meta['origin_date'][4:6]}-{meta['origin_date'][6:]}"
        # meta["origin_image"] = "_".join([comp for comp in meta["origin_image"].split("_")[:4] if not comp.isdigit()])
        meta_text = "N/A"  # '\n'.join(f"{k}: {v}" for k, v in meta.items())
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


if __name__ == "__main__":
    ds = PairedImageDataset("train_data_labelling.h5")

    count = 0

    for i in range(1000):
        try:
            if ds[i]["label"].max() > 0:
                count += 1
        except Exception as exc:
            print("Failed to show index", i, ":", exc)

    print(f"Count: {count} / {len(ds)}")
