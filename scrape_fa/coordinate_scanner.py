from __future__ import annotations

import click
import h5py
import glob
import json
import math
import numpy as np
import os
import rasterio
import yaml
from collections import defaultdict
from datetime import datetime, date
from PIL import Image
from rasterio.errors import RasterioIOError
from rasterio.warp import transform
from typing import Any

from scrape_fa.util.logging_config import setup_logging

LOGGER = setup_logging('coordinate_scanner')
WIDTH = None
HEIGHT = None


def group_coords(
    features: list[dict[str, Any]],
    step: float
) -> dict[tuple[float, float], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for feat in features:
        geom = feat.get('geometry', {})
        if geom.get('type') == 'Point':
            coords = geom.get('coordinates', [])
            if len(coords) == 2:
                lon, lat = coords
                # Group to nearest step
                lon_group = math.floor(lon / step) * step
                lat_group = math.floor(lat / step) * step
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        # Adjust grouping to include neighboring pixels
                        lon_group_adj = round(lon_group + i * step, 5)
                        lat_group_adj = round(lat_group + j * step, 5)
                        grouped[(lon_group_adj, lat_group_adj)].append(feat)
    return grouped


def process_group(
    src: rasterio.io.DatasetReader,
    feats: list[dict[str, Any]],
    lon: float,
    lat: float,
    step: float,
    origin_image: str,
    origin_date: str,
    src_crs: Any,
    wgs84_crs: Any
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # ToDo: make this caching less hacky
    global WIDTH, HEIGHT
    # Transform WGS84 (lon, lat) to raster CRS
    xs, ys = transform(wgs84_crs, src_crs, [lon - step, lon + 2 * step], [lat - step, lat + 2 * step])
    x1, x2 = xs[0], xs[1]
    y1, y2 = ys[0], ys[1]
    try:
        row1, col1 = src.index(x1, y1)
        row2, col2 = src.index(x2, y2)
        row_start, row_end = sorted([row1, row2])
        col_start, col_end = sorted([col1, col2])
        if WIDTH is None:
            WIDTH = col_end - col_start
        elif (col_end - col_start) != WIDTH:
            col_end = col_start + WIDTH
        if HEIGHT is None:
            HEIGHT = row_end - row_start
        elif (row_end - row_start) != HEIGHT:
            row_end = row_start + HEIGHT

        # Read RGB bands and convert to greyscale
        data = src.read([1, 2, 3], window=((row_start, row_end), (col_start, col_end)))
        if data.size == 0 or np.all(np.isnan(data)) or np.all(data == 0):
            return None, None, None
        r = data[0].astype(float)
        g = data[1].astype(float)
        b = data[2].astype(float)
        grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # Create label image
        label = np.zeros_like(grey, dtype=np.uint8)
        for feat in feats:
            feat_lon, feat_lat = feat['geometry']['coordinates']
            local_col = int(round((feat_lon - (lon - step)) / (3 * step) * (label.shape[1] - 1)))
            local_row = int(round((feat_lat - (lat - step)) / (3 * step) * (label.shape[0] - 1)))
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    rr = (label.shape[0] - local_row) + dr
                    cc = local_col + dc
                    if 0 <= rr < label.shape[0] and 0 <= cc < label.shape[1]:
                        label[rr, cc] = 255
        meta = {
            'origin_image': origin_image,
            'origin_date': origin_date,
            'lon_min': lon - step,
            'lon_max': lon + 2 * step,
            'lat_min': lat - step,
            'lat_max': lat + 2 * step
        }
        return grey.astype(np.float32), label, meta
    except Exception as e:
        LOGGER.exception("Failed to process group")
        return None, None, None


def parse_date_safe(date_str: str | None) -> date | None:
    """Parse YYYY-MM-DD or return None if missing/invalid."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def filter_tents_by_target_date(
    features: list[dict[str, Any]],
    date_target: date
) -> list[dict[str, Any]]:
    """Keep tents where start <= target <= end, or end is None."""
    filtered = []
    for feat in features:
        props = feat.get('properties', {})
        tent_start_dt = parse_date_safe(props.get('date_start'))
        tent_end_dt = parse_date_safe(props.get('date_end'))

        if tent_start_dt and tent_start_dt <= date_target:
            if tent_end_dt is None or date_target <= tent_end_dt:
                filtered.append(feat)
    return filtered


def extract_date_from_filename(filename: str) -> str | None:  # ### CHANGED: new helper
    """Extract first numeric sequence (YYYYMMDD) from a filename split by underscores."""
    parts = os.path.splitext(os.path.basename(filename))[0].split("_")
    for p in parts:
        if p.isdigit() and len(p) == 8:  # likely a date
            return p
    return None


def is_high_quality_tile(
    feats: list[dict[str, Any]],
    date_target_str: str,
    src: rasterio.io.DatasetReader,
    lon: float,
    lat: float,
    step: float,
    start_threshold: float,
    max_missing_end: float,
    min_valid_fraction: float
) -> bool:
    """
    Determine if a tile is high quality based on:
      1. Fraction of tents starting on the target date
      2. Fraction of tents with missing end date
      3. Completeness of raster data in the tile

    Args:
        feats (list): List of GeoJSON features in the tile.
        date_target_str (str): Date string from the filename (YYYYMMDD).
        src (rasterio.DatasetReader): Opened rasterio dataset.
        lon (float): Longitude of tile center.
        lat (float): Latitude of tile center.
        step (float): Tile step size.
        start_threshold (float): Minimum fraction of points with start date equal to target.
        max_missing_end (float): Maximum allowed fraction of tents with missing end date.
        min_valid_fraction (float): Minimum fraction of non-NaN, non-zero raster pixels.

    Returns:
        bool: True if tile meets all criteria, False otherwise.
    """
    if not feats or not date_target_str:
        return False

    # --- Date checks ---
    date_target = datetime.strptime(date_target_str, '%Y%m%d').date()
    start_matches = sum(parse_date_safe(f.get('properties', {}).get('date_start')) == date_target for f in feats)
    missing_end_count = sum(parse_date_safe(f.get('properties', {}).get('date_end')) is None for f in feats)

    start_fraction = start_matches / len(feats)
    missing_end_fraction = missing_end_count / len(feats)
    if start_fraction < start_threshold or missing_end_fraction > max_missing_end:
        return False

    # --- Raster data check ---
    try:
        # Transform tile bounds from WGS84 to raster CRS
        xs, ys = transform('EPSG:4326', src.crs, [lon - step, lon + 2 * step], [lat - step, lat + 2 * step])
        x1, x2 = xs[0], xs[1]
        y1, y2 = ys[0], ys[1]

        # Get pixel indices, clipped to raster bounds
        row1, col1 = src.index(x1, y1)
        row2, col2 = src.index(x2, y2)
        row_start, row_end = max(0, min(row1, row2)), min(src.height, max(row1, row2))
        col_start, col_end = max(0, min(col1, col2)), min(src.width, max(col1, col2))

        if row_start >= row_end or col_start >= col_end:
            return False  # Empty window

        data = src.read([1, 2, 3], window=((row_start, row_end), (col_start, col_end)))

        # Fraction of valid (non-NaN, non-zero) pixels
        valid_mask = ~np.isnan(data) & (data != 0)
        valid_fraction = np.count_nonzero(valid_mask) / data.size
        if valid_fraction < min_valid_fraction:
            return False
    except Exception:
        return False

    return True


class HDF5Writer:

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.file = h5py.File(self.hdf5_path, 'w')
        self.greyscale_group = self.file.create_group('feature')
        self.label_group = self.file.create_group('label')
        self.tile_idx = 0

    def add_entry(self, grey: np.ndarray, label: np.ndarray, meta: dict[str, Any]):
        tile_name = f"tile_{self.tile_idx:05d}"
        gset = self.greyscale_group.create_dataset(tile_name, data=grey, compression='gzip')
        lset = self.label_group.create_dataset(tile_name, data=label, compression='gzip')
        for ds in (gset, lset):
            ds.attrs['origin_image'] = meta['origin_image']
            ds.attrs['origin_date'] = meta['origin_date']
            ds.attrs['lon_min'] = round(meta['lon_min'], 5)
            ds.attrs['lon_max'] = round(meta['lon_max'], 5)
            ds.attrs['lat_min'] = round(meta['lat_min'], 5)
            ds.attrs['lat_max'] = round(meta['lat_max'], 5)
        self.tile_idx += 1

    def write(self):
        self.file.close()


def scan_grouped_coordinates(
    geotiff_path: str,
    geojson_path: str,
    hdf5_writer: 'HDF5Writer',
    quality_thresholds: dict[str, Any],
    step: float,
    date_target: str | None,
) -> None:
    try:
        src = rasterio.open(geotiff_path)
    except RasterioIOError:
        LOGGER.exception(f"Error opening GeoTIFF")
        return

    with open(geojson_path, 'r') as f:
        geojson = json.load(f)
        features = geojson.get('features', [])

    if date_target:
        try:
            date_obj = datetime.strptime(date_target, '%Y%m%d').date()
            features = filter_tents_by_target_date(features, date_obj)
            LOGGER.info(f"[{os.path.basename(geotiff_path)}] Filtered to {len(features)} features for date {date_target}.")
        except Exception:
            LOGGER.exception(f"Date parsing error for {geotiff_path}")
            return

    grouped = group_coords(features, step)
    LOGGER.info(f"[{os.path.basename(geotiff_path)}] Found {len(grouped)} coordinate groups with tents.")

    bounds = src.bounds
    LOGGER.info(f"GeoTIFF bounds: min_lon={bounds.left}, min_lat={bounds.bottom}, max_lon={bounds.right}, max_lat={bounds.top}")

    src_crs = src.crs
    wgs84_crs = 'EPSG:4326'

    high_quality_found = False  # Track if any HQ tiles are processed

    for (lon, lat), feats in grouped.items():
        if not is_high_quality_tile(feats, date_target, src, lon, lat, step, **quality_thresholds):
            continue  # skip low-quality tile
        high_quality_found = True
        grey, label, meta = process_group(src, feats, lon, lat, step, os.path.basename(geotiff_path), date_target or '', src_crs, wgs84_crs)
        if grey is not None and label is not None and meta is not None:
            hdf5_writer.add_entry(grey, label, meta)

    if not high_quality_found:
        LOGGER.warn(f"No valid high-quality tiles found in {os.path.basename(geotiff_path)}")

    src.close()


@click.command()
@click.argument('config', type=click.Path(exists=True, dir_okay=False))
def cli(config):
    """
    Run coordinate_scanner using a YAML config file.
    The YAML file must define: geotiff_dir, geojson, hdf5, step, start_threshold, max_missing_end, min_valid_fraction.
    Example YAML:
        geotiff_dir: path/to/geotiffs
        geojson: path/to/tents.geojson
        hdf5: path/to/output.h5
        step: 0.001
        start_threshold: 0.2
        max_missing_end: 0.2
        min_valid_fraction: 0.9
    """
    with open(config, 'r') as f:
        params = yaml.safe_load(f)
    required = ['geotiff_dir', 'geojson', 'hdf5', 'processing']
    for k in required:
        if k not in params:
            raise click.ClickException(f"Missing required config key: {k}")
    coordinate_scanner(
        params['geotiff_dir'],
        params['geojson'],
        params['hdf5'],
        **params['processing']
    )


def coordinate_scanner(geotiff_dir: str, geojson: str, hdf5: str, step: float, quality_thresholds: dict[str, Any]) -> None:
    tif_files = glob.glob(os.path.join(geotiff_dir, "*.tif"))
    if not tif_files:
        LOGGER.error(f"No .tif files found in {geotiff_dir}")
        return
    hdf5_writer = HDF5Writer(hdf5)
    for tif_path in tif_files:
        date_target = extract_date_from_filename(tif_path)
        if not date_target:
            LOGGER.warn(f"Skipping {tif_path} (no date found in filename).")
            continue

        LOGGER.info(f"Processing {tif_path} with date {date_target}...")
        scan_grouped_coordinates(tif_path, geojson, hdf5_writer, quality_thresholds, step, date_target)
    hdf5_writer.write()
    LOGGER.info(f"Saved dataset to {hdf5}")

if __name__ == "__main__":
    cli()

# EXAMPLE CLI USAGE:
# poetry run coordinate-scanner config.yaml
