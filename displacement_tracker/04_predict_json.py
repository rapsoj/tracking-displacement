import math
import yaml
import click
import torch
import rasterio
from rasterio.transform import from_bounds
from rasterio.merge import merge
from glob import glob
import numpy as np
import json
from pathlib import Path
from scipy.ndimage import label, center_of_mass

from displacement_tracker.paired_image_dataset import PairedImageDataset
from displacement_tracker.simple_cnn import SimpleCNN
from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.distance import interpolate_centroid
from displacement_tracker.util.deduplication import merge_close_points_global

LOGGER = setup_logging("predict_json")


def save_prediction_tiff(probs, bounds, out_path):
    """
    Save a single-band GeoTIFF from a probability array.
    probs: 2D numpy array (H, W)
    bounds: dict with lat_min, lat_max, lon_min, lon_max (may be in any order)
    """
    height, width = probs.shape

    # Defensive: coerce to floats and make sure min/max ordering is correct
    lat_min = float(bounds.get("lat_min"))
    lat_max = float(bounds.get("lat_max"))
    lon_min = float(bounds.get("lon_min"))
    lon_max = float(bounds.get("lon_max"))

    left = min(lon_min, lon_max)
    right = max(lon_min, lon_max)
    bottom = min(lat_min, lat_max)
    top = max(lat_min, lat_max)

    transform = from_bounds(left, bottom, right, top, width, height)

    with rasterio.open(
        str(out_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
        nodata=np.nan
    ) as dst:
        dst.write(probs.astype(np.float32), 1)


def merge_prediction_tiffs(tiff_dir, out_path, dst_crs="EPSG:4326"):
    """
    Merge all *_pred.tif files in tiff_dir into a single georeferenced GeoTIFF.
    """
    tiff_paths = sorted(glob(str(Path(tiff_dir) / "*_pred.tif")))
    if not tiff_paths:
        LOGGER.warning("No prediction TIFFs found to merge.")
        return

    src_files = [rasterio.open(p) for p in tiff_paths]
    mosaic, out_trans = merge(src_files)  # mosaic shape: (bands, H, W)

    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": dst_crs,
        "count": mosaic.shape[0],
        "dtype": mosaic.dtype
    })

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files:
        src.close()
    LOGGER.info(f"Mosaic written to {out_path}")



def predict(dataset, model, device, processing_cfg, sample_cfg=None, validation_tifs=False):
    """
    Run predictions and extract centroids of labeled regions above min_area for each tile.
    (No intra-tile spatial deduplication here; deduplication is performed globally after prediction.)
    """
    threshold = processing_cfg.get('threshold', 0.5)
    min_area = processing_cfg.get('min_area', 20)
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
                diff = entry["feature"] - entry["prewar"]
                feats = torch.cat((entry["feature"], entry["prewar"], diff), dim=0)
                feats = feats.unsqueeze(0).to(device)  # add batch dim
                outputs = model(feats)
                probs_np = outputs.cpu().squeeze().numpy()

                bounds = json.loads(entry["meta"])

                if crop_pixels > 0:
                    probs_np[:crop_pixels, :] = 0
                    probs_np[-crop_pixels:, :] = 0
                    probs_np[:, :crop_pixels] = 0
                    probs_np[:, -crop_pixels:] = 0

                if validation_tifs:
                    tiff_dir = Path(processing_cfg.get("tiff_output_dir", "prediction_tiffs"))
                    tiff_dir.mkdir(parents=True, exist_ok=True)

                    origin = bounds.get("origin_image", f"tile_{i}")
                    tiff_path = tiff_dir / f"{Path(origin).stem}_{i}_pred.tif"

                    save_prediction_tiff(probs_np, bounds, tiff_path)

                # Threshold
                mask = (probs_np > threshold)
                # Label regions
                labeled, num_features = label(mask)
                # Filter by min_area and extract centroids
                coords = []
                shape = mask.shape
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


def save_geojson(points, out_path):
    """
    Save results to a GeoJSON file.
    If deduplicated_points is provided, ONLY write those cleaned points (no original raw points,
    no deduplicated flag). Otherwise, fall back to writing per-tile polygons and points.
    """
    features = []

    # Write only cleaned/deduplicated points (one feature per merged centroid)
    for lat, lon in points:
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
        model_args={"n_channels": 3, "n_classes": 1}
    )
    validation_tifs = pred_cfg.get("validation_tifs", False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    sample_cfg = pred_cfg.get('sample', {})
    processing_cfg = pred_cfg.get('processing', {})
    out_path = pred_cfg.get('output', 'predictions.geojson')

    # run predictions (per-tile centroids, no intra-tile dedupe)
    results = predict(
        ds, model, device, processing_cfg, sample_cfg, validation_tifs=validation_tifs
    )
    flat_results = [pt for res in results for pt in res['coordinates']]
    LOGGER.info(f"Total number of tents (pre-merge): {len(flat_results)}")

    min_distance_m = processing_cfg.get('min_distance_m', 2.0)

    # Only keep points with agreement between overlaps
    agreement = processing_cfg.get("agreement", False)
    if isinstance(agreement, bool):  # convert to int, False -> 1 point agreement, True -> 2 point agreement
        agreement = 2 if agreement else 1

    # global deduplication in meters
    merged_coords = merge_close_points_global(
        flat_results,
        min_distance_m=min_distance_m,
        agreement=agreement
    )
    LOGGER.info(f"Total number of tents (post-merge): {len(merged_coords)}")

    # Save only the cleaned/deduplicated points (no raw points, no deduplicated flag)
    save_geojson(merged_coords, out_path)

    if validation_tifs:
        tiff_dir = Path(processing_cfg.get("tiff_output_dir", "prediction_tiffs"))
        mosaic_out = Path(pred_cfg.get("tiff_mosaic_output", "predictions_mosaic.tif"))
        merge_prediction_tiffs(tiff_dir, str(mosaic_out))

if __name__ == '__main__':
    cli()
