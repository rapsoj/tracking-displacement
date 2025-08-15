import os
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, folder, feat_transform=None, label_transform=None):

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
        self.folder = folder
        self.feat_transform = feat_transform
        self.label_transform = label_transform
        self.pairs = self._find_pairs()

    def _find_pairs(self):
        files = os.listdir(self.folder)
        feat_files = [f for f in files if f.endswith('_feat.png')]
        label_files = [f for f in files if f.endswith('_label.png')]
        # Map by base name (without _feat/_label)
        feat_map = {f.replace('_feat.png', ''): f for f in feat_files}
        label_map = {f.replace('_label.png', ''): f for f in label_files}
        # Only keep pairs that exist in both
        common_keys = set(feat_map.keys()) & set(label_map.keys())
        filtered_pairs = []
        for k in sorted(common_keys):
            feat_path = os.path.join(self.folder, feat_map[k])
            label_path = os.path.join(self.folder, label_map[k])
            # Check label image is not all black
            with Image.open(label_path) as label_img:
                label_arr = np.array(label_img)
                if np.all(label_arr == 0):
                    continue  # skip if label is all black
            # Check feature image for >5% black pixels
            with Image.open(feat_path) as feat_img:
                feat_arr = np.array(feat_img)
                total_pixels = feat_arr.size
                black_pixels = np.sum(feat_arr == 0)
                if black_pixels / total_pixels > 0.05:
                    continue  # skip if >5% of feature is black
            filtered_pairs.append((k, feat_map[k], label_map[k]))
        return filtered_pairs

    def _get_min_label_size(self):
        min_w, min_h = None, None
        for k, _, label_name in self.pairs:
            label_path = os.path.join(self.folder, label_name)
            with Image.open(label_path) as img:
                w, h = img.size
                if min_w is None or w < min_w:
                    min_w = w
                if min_h is None or h < min_h:
                    min_h = h
        # Eliminate left and bottom row of pixels
        min_w = max(1, min_w - 1)
        min_h = max(1, min_h - 1)
        return min_w, min_h

    def __len__(self):
        # 8 augmentations per pair (4 rotations x 2 flips)
        return len(self.pairs)

    def __getitem__(self, idx):
        # Determine which pair and which augmentation
        k, feat_name, label_name = self.pairs[idx]
        feat_path = os.path.join(self.folder, feat_name)
        label_path = os.path.join(self.folder, label_name)
        feat_img = Image.open(feat_path).convert('L')
        label_img = Image.open(label_path).convert('L')
        # Crop to smallest label size, eliminating left and bottom row
        if not hasattr(self, '_min_label_size'):
            self._min_label_size = self._get_min_label_size()
        min_w, min_h = self._min_label_size
        feat_img = feat_img.crop((1, 0, 1 + min_w, min_h))
        label_img = label_img.crop((1, 0, 1 + min_w, min_h))
        if self.feat_transform:
            feat_img = self.feat_transform(feat_img)
        else:
            feat_img = torch.from_numpy(np.array(feat_img)).float().unsqueeze(0) / 255.0
        if self.label_transform:
            label_img = self.label_transform(label_img)
        else:
            label_img = torch.from_numpy(np.array(label_img)).float().unsqueeze(0) / 255.0
        return feat_img, label_img

# Example usage:
# dataset = PairedImageDataset('/path/to/folder')
# img, label = dataset[0]
