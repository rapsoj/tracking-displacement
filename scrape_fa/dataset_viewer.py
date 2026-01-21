import ast
import torch
import numpy as np
import matplotlib.pyplot as plt

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
            meta_dict = ast.literal_eval(str(meta))
            meta_text = '\n'.join(f"{k}: {v}" for k, v in meta_dict.items())
        except (ValueError, SyntaxError):
            meta_text = str(meta)

        return arr_feat, arr_label, meta_text

    def _plot_meta(self, ax, text):
        ax.axis('off')
        ax.text(0.1, 0.9, text, fontsize=12, color='black',
                ha='left', va='top', transform=ax.transAxes)

    def show_overlay(self, idx: int) -> None:
        arr_feat, arr_label, meta_text = self._prepare_display_data(idx)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 6]})
        self._plot_meta(axes[0], meta_text)

        axes[1].imshow(arr_feat, cmap='gray', interpolation='none')
        axes[1].set_title('Data')
        axes[1].axis('off')

        axes[1].imshow(np.ones_like(arr_label), cmap="spring", alpha=arr_label, interpolation='none')

        # Draw grid lines
        h, w = arr_label.shape
        for i in range(1, 3):
            axes[1].axhline(y=i * h // 3, color='red', linestyle='--')
            axes[1].axvline(x=i * w // 3, color='red', linestyle='--')

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

if __name__ == "__main__":
    from scrape_fa.paired_image_dataset import PairedImageDataset
    ds = PairedImageDataset("test_data_labelling.h5")

    n_non_null, counts = np.unique(np.nonzero(ds.label_dataset)[0], return_counts=True)
    sorting = np.argsort(counts)
    counts = counts[sorting] // 9
    counts[counts == 0] = 1
    n_non_null = n_non_null[sorting]

    viewer = DatasetViewer(ds)
    viewer.show_overlay(n_non_null[-10])
