import math
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

EARTH_RADIUS_M = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Haversine distance in meters between two (lat, lon) pairs.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra = self.find(a); rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

def merge_close_points_global(results, min_distance_m=2.0):
    """
    Merge points across all tiles that are closer than min_distance_m.
    Returns a flat list of merged (lat, lon) centroids.
    """
    # Flatten all points
    flat = []  # (lat, lon, tile_idx, pt_idx)
    for t_idx, tile in enumerate(results):
        for p_idx, (lat, lon) in enumerate(tile.get('coordinates', [])):
            flat.append((lat, lon, t_idx, p_idx))
    n = len(flat)
    if n == 0:
        return []

    uf = UnionFind(n)

    # O(n^2) pairwise merge; OK for moderate n (few thousands).
    for i in range(n):
        lat_i, lon_i, _, _ = flat[i]
        for j in range(i+1, n):
            lat_j, lon_j, _, _ = flat[j]
            if haversine_m(lat_i, lon_i, lat_j, lon_j) <= min_distance_m:
                uf.union(i, j)

    # collect clusters
    clusters = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)

    # compute centroid for each cluster (simple average of lat/lon)
    merged = []
    for members in clusters.values():
        sum_lat = 0.0
        sum_lon = 0.0
        for m in members:
            lat, lon, _, _ = flat[m]
            sum_lat += lat
            sum_lon += lon
        cnt = len(members)
        merged.append((sum_lat / cnt, sum_lon / cnt))
    return merged

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
    (No intra-tile spatial deduplication here; deduplication is performed globally after prediction.)
    """
    threshold = processing_cfg.get('threshold', 0.5)
    min_area = processing_cfg.get('min_area', 20)

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
                            # no intra-tile duplicate removal here; do global merge later
                            coords.append(coord)
                        except Exception as exc:
                            LOGGER.warning(f"Interpolation error: {exc}")
                results.append({
                    "bounds": bounds,
                    'coordinates': coords
                })
                LOGGER.info(f"Found {len(coords)} tents (pre-merge).")
            except Exception as exc:
                LOGGER.warning(f"Prediction error: {exc}")
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

def save_geojson(results, out_path, save_bounds: bool = True, deduplicated_points=None):
    """
    Save results to a GeoJSON file.
    If deduplicated_points is provided, ONLY write those cleaned points (no original raw points,
    no deduplicated flag). Otherwise, fall back to writing per-tile polygons and points.
    """
    features = []

    if deduplicated_points:
        # Write only cleaned/deduplicated points (one feature per merged centroid)
        for lat, lon in deduplicated_points:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "name": "tents"
                }
            })
    else:
        # legacy behaviour: write polygons and per-tile points
        for tile_idx, tile in enumerate(results):
            coords = tile.get('coordinates', [])
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
                        "date_start": (tile.get('bounds') or {}).get('origin_date'),
                        "name": "tents",
                        "date_end": (tile.get('bounds') or {}).get('origin_date'),
                        "source": (tile.get('bounds') or {}).get("origin_image")
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

    # run predictions (per-tile centroids, no intra-tile dedupe)
    results = predict_json(ds, model, device, processing_cfg, sample_cfg)
    tent_count_pre = sum(len(res["coordinates"]) for res in results)
    LOGGER.info(f"Total number of tents (pre-merge): {tent_count_pre}")

    # global deduplication in meters
    min_distance_m = processing_cfg.get('min_distance_m', 2.0)
    merged_coords = merge_close_points_global(results, min_distance_m=min_distance_m)
    LOGGER.info(f"Total number of tents (post-merge): {len(merged_coords)}")

    # Save only the cleaned/deduplicated points (no raw points, no deduplicated flag)
    save_geojson(results, out_path, save_bounds=False, deduplicated_points=merged_coords)

if __name__ == '__main__':
    cli()
