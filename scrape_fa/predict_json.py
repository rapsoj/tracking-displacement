import yaml
import click
import torch
import random
import numpy as np
import json
from pathlib import Path
from scipy.ndimage import label, center_of_mass

from scrape_fa.paired_image_dataset import PairedImageDataset
from scrape_fa.simple_cnn import SimpleCNN
from scrape_fa.util.logging_config import setup_logging

LOGGER = setup_logging("predict_json")

def interpolate_centroid(centroid, bounds, shape):
    """
    Convert centroid pixel coordinates to geographic coordinates using bounds.
    centroid: (y, x) pixel coordinates
    bounds: dict with lat_min, lat_max, lon_min, lon_max
    shape: (height, width) of the image
    Returns (latitude, longitude)
    """
    y, x = centroid
    height, width = shape
    lat_min = bounds.get('lat_min')
    lat_max = bounds.get('lat_max')
    lon_min = bounds.get('lon_min')
    lon_max = bounds.get('lon_max')
    if None in (lat_min, lat_max, lon_min, lon_max):
        raise ValueError("Missing bounds in metadata")
    lat = lat_min + (lat_max - lat_min) * (y / height)
    lon = lon_min + (lon_max - lon_min) * (x / width)
    return (lat, lon)

def predict_json(dataset, model, device, processing_cfg, sample_cfg=None):
    """
    Run predictions and extract centroids of labeled regions above min_area for each tile.
    """
    threshold = processing_cfg.get('threshold', 0.5)
    min_area = processing_cfg.get('min_area', 20)
    label_value = processing_cfg.get('label', 1)

    if sample_cfg and sample_cfg.get('enable', False):
        total = len(dataset)
        size = min(sample_cfg.get('size', total), total)
        seed = sample_cfg.get('seed', None)
        frac = float(size) / float(total)
        (subset, ), (idcs, ) = dataset.create_subset([frac, 1-frac], seed=seed)
        LOGGER.info(f"🔹 Using random sample of {size}/{total} tiles for prediction.")
    else:
        subset = dataset
        LOGGER.info(f"🔹 Using all {len(dataset)} tiles for prediction.")

    results = []
    with torch.no_grad():
        for i, entry in enumerate(subset):
            try:
                LOGGER.info(f"Predicting image {i+1}/{len(dataset)}")
                feats = torch.cat((entry["feature"], entry["prewar"]))
                feats = feats.to(device)
                outputs = model(feats)
                # Apply sigmoid
                probs = torch.sigmoid(outputs)
                probs_np = probs.cpu().squeeze().numpy()
                # Threshold
                mask = (probs_np > threshold).astype(np.uint8)
                # Label regions
                labeled, num_features = label(mask == label_value)
                # Filter by min_area and extract centroids
                coords = []
                shape = mask.shape
                bounds = entry['meta']
                for region_id in range(1, num_features + 1):
                    region_mask = (labeled == region_id)
                    area = np.sum(region_mask)
                    if area > min_area:
                        centroid = center_of_mass(region_mask)
                        try:
                            coord = interpolate_centroid(centroid, bounds, shape)
                            coords.append(coord)
                        except Exception as exc:
                            LOGGER.warning(f"Interpolation error: {exc}")
                results.append({
                    "bounds": bounds,
                    'coordinates': coords
                })
            except Exception as exc:
                LOGGER.warning(f"Prediction error: {exc}")
    # Wait for further instructions; do not save or process further
    return results

def bounds_to_polygon(bounds):
    """
    Convert bounds dict to a GeoJSON polygon (list of [lon, lat] pairs).
    """
    lat_min = bounds.get('lat_min')
    lat_max = bounds.get('lat_max')
    lon_min = bounds.get('lon_min')
    lon_max = bounds.get('lon_max')
    if None in (lat_min, lat_max, lon_min, lon_max):
        raise ValueError("Missing bounds in metadata")
    # Polygon corners: lower left, lower right, upper right, upper left, close
    return [
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min]
    ]

def save_geojson(results, out_path):
    """
    Save results to a GeoJSON file. Each tile's bounds as Polygon, centroids as Points.
    """
    features = []
    for tile_idx, tile in enumerate(results):
        coords = tile['coordinates']
        bounds = tile.get('bounds')
        if bounds:
            try:
                polygon = bounds_to_polygon(bounds)
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon]
                    },
                    "properties": {
                        "name": "tents",
                        "date_start": bounds.get('origin_date'),
                        "date_end": bounds.get('origin_date'),
                        "source": bounds.get("origin_image")
                    }
                })
            except Exception as exc:
                LOGGER.warning(f"Polygon conversion error: {exc}")
        for pt_idx, (lat, lon) in enumerate(coords):
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "tile_index": tile_idx,
                    "centroid_index": pt_idx
                }
            })
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    with open(out_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    LOGGER.info(f"GeoJSON saved to {out_path}")

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
    processing_cfg = pred_cfg.get('processing', {})
    out_path = pred_cfg.get('output', 'predictions.geojson')
    results = predict_json(ds, model, device, processing_cfg, sample_cfg)
    save_geojson(results, out_path)

if __name__ == '__main__':
    cli()
