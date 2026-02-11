from __future__ import annotations
import random
from pathlib import Path
from functools import lru_cache

from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter

from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("paired-image-ds")


class PairedImageDataset(Dataset):
    def __init__(self, hdf5_path, indices: list[int] | None = None, sigma: float = 3.0):
        self.hdf5_path = hdf5_path
        self.indices: list | None = indices
        self.sigma = sigma

        # Calculate normalization constant dynamically
        # Create dummy image: 100x100 black image with 3x3 white square in center
        dummy = np.zeros((100, 100), dtype=np.float32)
        dummy[49:52, 49:52] = 255.0
        blurred_dummy = gaussian_filter(dummy, sigma=self.sigma)
        # Rescaled constant so that value of 1 is never exceeded, even with multiple tents together
        self.norm_constant = float(np.max(blurred_dummy)) * 2.3

        LOGGER.info(
            f"Initialized PairedImageDataset with sigma={self.sigma}. Normalization constant: {self.norm_constant}"
        )

        # Default feature transform: numpy array to tensor

        self.h5 = h5py.File(self.hdf5_path, "r")  # r+ supports saving of splits
        self.greyscale_dataset = self.h5["feature"]
        self.prewar_dataset = self.h5["prewar"]
        self.label_dataset = self.h5["label"]
        self.meta_dataset = self.h5["meta"]

    @staticmethod
    def feat_transform(arr):
        return torch.from_numpy(arr).unsqueeze(0)

    def label_transform(self, arr):
        arr = arr.astype(np.float32)
        blurred = gaussian_filter(arr, sigma=self.sigma)
        return torch.from_numpy(blurred).unsqueeze(0) / self.norm_constant

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
        feat = PairedImageDataset.feat_transform(grey)
        lab = self.label_transform(label_arr)
        prewar = PairedImageDataset.feat_transform(prewar)
        return {"feature": feat, "label": lab, "meta": meta, "prewar": prewar}

    def label_is_negative(self, i):
        sample = self[i]
        label = sample["label"]
        return (label.max() == 0).item()

    def create_subsets(
        self,
        splits: list[float],
        shuffle: bool = True,
        save_loc: str | None = None,
        regenerate_splits: bool = False,
        seed: int | None = None,
    ) -> list["PairedImageDataset"]:
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
            if not cache_valid:  # This means we didn't cache them
                idcs_list.append(subset_indices)
            else:
                subset_indices = idcs_list[i]
            datasets.append(
                PairedImageDataset(
                    self.hdf5_path, indices=subset_indices, sigma=self.sigma
                )
            )

            start_idx = end_idx

        return datasets, idcs_list

    def close(self):
        self.h5.close()
