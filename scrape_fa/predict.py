import yaml
import click
import torch
import random

from scrape_fa.paired_image_dataset import PairedImageDataset
from scrape_fa.simple_cnn import SimpleCNN
from scrape_fa.util.logging_config import setup_logging

LOGGER = setup_logging("predict")

def prediction(dataset, model, target_dir, device, sample_cfg=None):
    """
    Run predictions on dataset (optionally subsampling tiles).
    """
    if sample_cfg and sample_cfg.get('enable', False):
        total = len(dataset)
        size = min(sample_cfg.get('size', total), total)
        seed = sample_cfg.get('seed', None)
        if seed is not None:
            random.seed(seed)
        indices = random.sample(range(total), size)
        dataset = dataset.__class__(
            dataset.hdf5_path if hasattr(dataset, "hdf5_path") else dataset.path,
            indices=indices,
            feat_transform=getattr(dataset, "feat_transform", None),
            label_transform=getattr(dataset, "label_transform", None),
            is_pred=getattr(dataset, "is_pred", False)
        )
        LOGGER.info(f"🔹 Using random sample of {size}/{total} tiles for prediction.")
    else:
        LOGGER.info(f"🔹 Using all {len(dataset)} tiles for prediction.")

    output_combined = []
    with torch.no_grad():
        for i, entry in enumerate(dataset):
            try:
                LOGGER.info(f"Predicting image {i+1}/{len(dataset)}")
                feats = torch.cat((entry["feature"], entry["prewar"]))
                feats = feats.to(device)
                outputs = model(feats)
                output_combined.append(outputs.cpu().squeeze())
            except Exception as exc:
                LOGGER.warning(f"Prediction error: {exc}")

    PairedImageDataset.from_predictions(dataset, output_combined, target_dir)


@click.command()
@click.argument('config', type=click.Path(exists=True))
def cli(config) -> None:
    with open(config, 'r') as f:
        params = yaml.safe_load(f)

    if 'prediction' not in params:
        raise click.ClickException("Missing required config key: prediction")

    pred_cfg = params['prediction']
    ds = PairedImageDataset(pred_cfg['input'])
    model = SimpleCNN.from_pth(
        pred_cfg['model'],
        model_args={"n_channels": 2, "n_classes": 1}
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    sample_cfg = pred_cfg.get('sample', {})
    prediction(ds, model, pred_cfg['output'], device, sample_cfg)


if __name__ == '__main__':
    cli()
