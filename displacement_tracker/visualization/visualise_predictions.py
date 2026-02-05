import h5py
import numpy as np
import matplotlib.pyplot as plt
import yaml
import click

plt.ion()  # enable interactive mode


class TileViewer:
    def __init__(self, processed_h5_path, predictions_h5_path):
        self.processed_h5 = h5py.File(processed_h5_path, "r")
        self.predictions_h5 = h5py.File(predictions_h5_path, "r")
        self.tiles = list(self.processed_h5["feature"].keys())
        self.index = 0

        self.fig, self.axes = plt.subplots(1, 4, figsize=(20, 5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update()

    def overlay(self, base, overlay, alpha=0.5, color="red"):
        """Overlay a mask on the base image with specified color"""
        base_rgb = np.stack([base] * 3, axis=-1)
        overlay_rgb = np.zeros_like(base_rgb)
        channel = {"red": 0, "green": 1, "blue": 2}[color]
        overlay_rgb[..., channel] = overlay
        overlay_norm = (
            overlay_rgb / overlay_rgb.max() if overlay_rgb.max() > 0 else overlay_rgb
        )
        return (1 - alpha) * base_rgb + alpha * overlay_norm * 255

    def update(self):
        tile_name = self.tiles[self.index]

        current = self.processed_h5["feature"][tile_name][()]
        prewar = self.processed_h5["prewar"][tile_name][()]
        label = self.processed_h5["label"][tile_name][()]
        pred = self.predictions_h5["predictions"][tile_name][()]

        # ensure 2D images
        if current.ndim > 2:
            current = current[0]
        if prewar.ndim > 2:
            prewar = prewar[0]
        if label.ndim > 2:
            label = label[0]

        self.axes[0].imshow(prewar, cmap="gray")
        self.axes[0].set_title("Pre-war")
        self.axes[0].axis("off")

        self.axes[1].imshow(current, cmap="gray")
        self.axes[1].set_title("Current")
        self.axes[1].axis("off")

        self.axes[2].imshow(self.overlay(current, pred, color="red"))
        self.axes[2].set_title("Prediction Overlay")
        self.axes[2].axis("off")

        self.axes[3].imshow(self.overlay(current, label, color="green"))
        self.axes[3].set_title("Label Overlay")
        self.axes[3].axis("off")

        self.fig.suptitle(f"Tile {self.index + 1}/{len(self.tiles)}: {tile_name}")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "right":
            self.index = (self.index + 1) % len(self.tiles)
            self.update()
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.tiles)
            self.update()


@click.command()
@click.argument("config", type=click.Path(exists=True))
def cli(config):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    pred_cfg = cfg.get("prediction", {})
    processed_path = pred_cfg.get("input")
    predictions_path = pred_cfg.get("output")
    if not processed_path or not predictions_path:
        raise click.ClickException(
            "Config must specify prediction/input and prediction/output paths"
        )

    viewer = TileViewer(processed_path, predictions_path)
    plt.show(block=True)


if __name__ == "__main__":
    cli()
