import yaml
import click
import torch
import random
import numpy as np
import json
from pathlib import Path
from scipy.ndimage import label, center_of_mass
#import matplotlib
#matplotlib.use("agg")
#import matplotlib.pyplot as plt

from scrape_fa.paired_image_dataset import PairedImageDataset
from scrape_fa.simple_cnn import SimpleCNN
from scrape_fa.util.logging_config import setup_logging

LOGGER = setup_logging("predict_json")

def too_close_criterion(pt1: tuple[float, float], pt2: tuple[float, float], min_distance: float) -> bool:
    lat_1, lon_1 = pt1
    lat_2, lon_2 = pt2

    dist_lat = lat_1 - lat_2
    dist_lon = lon_1 - lon_2
    distance = (dist_lat**2 + dist_lon**2)**0.5

    return distance < min_distance

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
    lat = lat_max - (lat_max - lat_min) * (y / height)
    lon = lon_min + (lon_max - lon_min) * (x / width)
    return (lat, lon)

def predict_json(dataset, model, device, processing_cfg, sample_cfg=None):
    """
    Run predictions and extract centroids of labeled regions above min_area for each tile.
    """
    threshold = processing_cfg.get('threshold', 0.5)
    min_area = processing_cfg.get('min_area', 20)
    min_distance = processing_cfg.get('min_distance', 0.0001)
    crop_pixels = processing_cfg.get('crop_pixels', 0)


    if sample_cfg and sample_cfg.get('enable', True):
        total = len(dataset)
        size = min(sample_cfg.get('size', total), total)
        seed = sample_cfg.get('seed', None)
        frac = float(size) / float(total)
        subsets, _ = dataset.create_subsets([frac, 1-frac], seed=seed)
        subset = subsets[0]
        LOGGER.info(f"🔹 Using random sample of {size}/{total} tiles for prediction.")
    else:
        subset = dataset
        LOGGER.info(f"🔹 Using all {len(dataset)} tiles for prediction.")

    results = []
    with torch.no_grad():
        for i, entry in enumerate(subset):
            try:
                LOGGER.info(f"Predicting image {i+1}/{len(subset)}")
                feats = torch.cat((entry["feature"], entry["prewar"]))
                feats = feats.to(device)
                outputs = model(feats)
                probs_np = outputs.cpu().squeeze().numpy()

                if crop_pixels > 0:
                    probs_np[:crop_pixels, :] = 0
                    probs_np[-crop_pixels:, :] = 0
                    probs_np[:, :crop_pixels] = 0
                    probs_np[:, -crop_pixels:] = 0

                # Threshold
                mask = (probs_np > threshold)
                # Label regions
                labeled, num_features = label(mask)
                # Filter by min_area and extract centroids
                coords = []
                shape = mask.shape
                bounds = json.loads(entry['meta'])
                for region_id in range(1, num_features + 1):
                    region_mask = (labeled == region_id)
                    area = np.sum(region_mask)
                    if area > min_area:
                        centroid = center_of_mass(region_mask)
                        try:
                            coord = interpolate_centroid(centroid, bounds, shape)
                            any_too_close = any(
                                too_close_criterion(
                                    coord,
                                    existing,
                                    min_distance=min_distance
                                ) for existing in coords
                            )
                            if not any_too_close:
                                coords.append(coord)
                        except Exception as exc:
                            LOGGER.warning(f"Interpolation error: {exc}")
                results.append({
                    "bounds": bounds,
                    'coordinates': coords
                })
                LOGGER.info(f"Found {len(coords)} tents.")
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

def save_geojson(results, out_path, save_bounds: bool = True):
    """
    Save results to a GeoJSON file. Each tile's bounds as Polygon, centroids as Points.
    """
    features = []
    for tile_idx, tile in enumerate(results):
        coords = tile['coordinates']
        bounds = tile.get('bounds')
        if bounds and save_bounds:
            try:
                polygon = bounds_to_polygon(bounds)
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon]
                    },
                    "properties": {
                        "name": "region_processed",
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
                    "date_start": bounds.get('origin_date'),
                    "name": "tents",
                    "date_end": bounds.get('origin_date'),
                    "source": bounds.get("origin_image")
                }
            })
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    with open(out_path, 'w', encoding="utf-8") as f:
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
    tent_count = sum(len(res["coordinates"]) for res in results)
    LOGGER.info(f"Total number of tents: {tent_count}")
    save_geojson(results, out_path, pred_cfg.get("save_bounds", True))

if __name__ == '__main__':
    cli()
