import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

class DatasetViewer:
    def __init__(self, dataset):
        """
        :param dataset: The dataset instance (e.g., PairedImageDataset) to view.
        """
        self.dataset = dataset

    def show(self, idx: int, overlay: bool = True) -> None:
        if overlay:
            self.show_overlay(idx)
        else:
            self.show_split(idx)

    def _prepare_display_data(self, idx: int):
        sample = self.dataset[idx]

        def to_numpy(t):
            return t.squeeze().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)

        arr_feat = to_numpy(sample['feature'])
        arr_label = to_numpy(sample['label'])

        meta = sample['meta']
        if isinstance(meta, bytes):
            meta = meta.decode('utf-8')

        try:
            meta_dict = json.loads(str(meta))
            meta_text = '\n'.join(f"{k}: {v}" for k, v in meta_dict.items())
        except json.JSONDecodeError:
            meta_text = str(meta)

        return arr_feat, arr_label, meta_text

    def _plot_meta(self, ax, text):
        ax.axis('off')
        ax.text(0.1, 0.9, text, fontsize=12, color='black',
                ha='left', va='top', transform=ax.transAxes)

    def _plot_overlay_on_axis(self, ax, arr_feat, arr_label):
        ax.imshow(arr_feat, cmap='gray', interpolation='none')
        ax.imshow(np.ones_like(arr_label), cmap="spring", alpha=arr_label, interpolation='none')

        # Draw grid lines
        h, w = arr_label.shape
        for i in range(1, 3):
            ax.axhline(y=i * h // 3, color='red', linestyle='--')
            ax.axvline(x=i * w // 3, color='red', linestyle='--')
        ax.axis('off')

    def show_overlay(self, idx: int) -> None:
        arr_feat, arr_label, meta_text = self._prepare_display_data(idx)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 6]})
        self._plot_meta(axes[0], meta_text)

        self._plot_overlay_on_axis(axes[1], arr_feat, arr_label)
        axes[1].set_title('Data')

        plt.tight_layout()
        plt.show()

    def show_split(self, idx: int) -> None:
        arr_feat, arr_label, meta_text = self._prepare_display_data(idx)

        fig, axes = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 3, 3]})
        self._plot_meta(axes[0], meta_text)

        axes[1].imshow(arr_feat, cmap='gray', interpolation='none')
        axes[1].set_title('Feature')
        axes[1].axis('off')

        axes[2].imshow(arr_label, cmap='gray')
        axes[2].set_title('Label')
        axes[2].axis('off')
        plt.tight_layout()
        plt.show()

    def show_batch(self, indices: list[int]) -> None:
        indices = indices[:18]
        n_plots = len(indices)
        if n_plots == 0:
            return

        # Calculate optimal layout to prioritize even split (minimal empty tiles), then max scale
        best_rows = 1
        best_cols = n_plots
        best_waste = float('inf')
        best_scale = 0.0

        for cols in range(1, min(n_plots, 6) + 1):
            rows = (n_plots + cols - 1) // cols
            waste = rows * cols - n_plots

            # Calculate potential scale (size of square tile) allowed by width (12) and height (8) limits
            scale = min(12 / cols, 8 / rows)

            # Priority 1: Minimize waste (empty tiles)
            # Priority 2: Maximize scale (image size)
            if waste < best_waste:
                best_waste = waste
                best_rows, best_cols = rows, cols
                best_scale = scale
            elif waste == best_waste:
                if scale > best_scale:
                    best_rows, best_cols = rows, cols
                    best_scale = scale

        figsize = (best_cols * best_scale, best_rows * best_scale)
        fig, axes = plt.subplots(best_rows, best_cols, figsize=figsize)

        # Ensure axes is iterable even for single plot
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for i, idx in enumerate(indices):
            ax = axes_flat[i]
            arr_feat, arr_label, _ = self._prepare_display_data(idx)

            self._plot_overlay_on_axis(ax, arr_feat, arr_label)

            txt = ax.text(0.5, 0.95, str(idx), fontsize=10, color='lightgreen',
                    ha='center', va='top', transform=ax.transAxes, fontweight='bold')
            txt.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])

        # Hide unused subplots
        for i in range(n_plots, len(axes_flat)):
            axes_flat[i].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    from scrape_fa.paired_image_dataset import PairedImageDataset
    ds = PairedImageDataset("test_data_labelling.h5")

    n_non_null, counts = np.unique(np.nonzero(ds.label_dataset)[0], return_counts=True)
    sorting = np.argsort(counts)
    counts = counts[sorting] // 9
    counts[counts == 0] = 1
    n_non_null = n_non_null[sorting]

    viewer = DatasetViewer(ds)
    viewer.show_batch(n_non_null[-15:])