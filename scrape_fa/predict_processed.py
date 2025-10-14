import h5py
import torch
import numpy as np
from tqdm import tqdm
import click
import yaml
from scrape_fa.simple_cnn import SimpleCNN  # your model
from scrape_fa.util.logging_config import setup_logging

LOGGER = setup_logging("predict_processed")


def predict_processed(h5_path, model_path, output_path, device="cuda", sample_cfg=None):
    """
    Run predictions on tiles in an HDF5 dataset and save outputs.
    Can optionally sample a subset of tiles.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    LOGGER.info(f"Loading model from {model_path}")
    model = SimpleCNN.from_pth(model_path, model_args={"n_channels": 2, "n_classes": 1})
    model.to(device)
    model.eval()

    with h5py.File(h5_path, "r") as infile, h5py.File(output_path, "w") as outfile:
        features = infile["feature"]
        tile_names = list(features.keys())

        # Apply random sampling if requested
        if sample_cfg and sample_cfg.get("enable", False):
            import random

            seed = sample_cfg.get("seed", None)
            if seed is not None:
                random.seed(seed)
            size = min(sample_cfg.get("size", len(tile_names)), len(tile_names))
            tile_names = random.sample(tile_names, size)
            LOGGER.info(f"Using random sample of {size} tiles")

        out_group = outfile.create_group("predictions")

        for tile_name in tqdm(tile_names, desc="Predicting tiles"):
            data = features[tile_name][()]
            # Ensure proper shape: (1, H, W)
            if data.ndim == 2:
                data = np.expand_dims(data, axis=0)
            # For paired data (feature + prewar), concatenate if available
            if "prewar" in infile:
                prewar = infile["prewar"][tile_name][()]
                if prewar.ndim == 2:
                    prewar = np.expand_dims(prewar, axis=0)
                data = np.concatenate([data, prewar], axis=0)

            tensor = torch.from_numpy(data).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred = model(tensor)
                pred_np = torch.sigmoid(pred).cpu().squeeze().numpy()

            # Save prediction
            out_group.create_dataset(tile_name, data=pred_np, compression="gzip")
            # Copy attributes
            for attr, val in features[tile_name].attrs.items():
                out_group[tile_name].attrs[attr] = val

    LOGGER.info(f"Saved predictions to {output_path}")


@click.command()
@click.option("--config", type=click.Path(exists=True), help="YAML config file")
@click.option("--input", type=click.Path(exists=True), help="HDF5 input path (overrides config)")
@click.option("--model", type=click.Path(exists=True), help="Model .pth path (overrides config)")
@click.option("--output", type=click.Path(), help="Output HDF5 path (overrides config)")
@click.option("--device", default="cuda", help="Device (cuda or cpu)")
def cli(config, input, model, output, device):
    """
    Predict on processed_data.h5 using a YAML config or direct paths.
    """
    sample_cfg = None

    if config:
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
        pred_cfg = cfg.get("prediction", {})
        input = input or pred_cfg.get("input")
        model = model or pred_cfg.get("model")
        output = output or pred_cfg.get("output")
        sample_cfg = pred_cfg.get("sample", None)

    if not (input and model and output):
        raise click.ClickException("Must provide input, model, and output paths")

    predict_processed(input, model, output, device, sample_cfg)


if __name__ == "__main__":
    cli()
