import rasterio
import json
import numpy as np
import math
from datetime import datetime, timedelta
from collections import defaultdict
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds
import os
import glob
from PIL import Image
import click
from tqdm import tqdm


def save_greyscale_only(grey, lon, lat, step, out_dir, date_target_str=None):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{lon:.5f}_{lat:.5f}_{date_target_str}_feat.png" if date_target_str else f"{lon:.5f}_{lat:.5f}_feat.png"
    out_path = os.path.join(out_dir, fname)
    img = Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8))
    img.save(out_path)


def save_greyscale_and_label(grey, feats, lon, lat, step, out_dir, date_target_str=None):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{lon:.5f}_{lat:.5f}_{date_target_str}_feat.png"
    out_path = os.path.join(out_dir, fname)
    img = Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8))
    img.save(out_path)

    # Create label image
    label = np.zeros_like(grey, dtype=np.uint8)
    for feat in feats:
        feat_lon, feat_lat = feat['geometry']['coordinates']
        local_col = int(round((feat_lon - (lon - step)) / (3 * step) * (label.shape[1] - 1)))
        local_row = int(round((feat_lat - (lat - step)) / (3 * step) * (label.shape[0] - 1)))
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr = local_row + dr
                cc = local_col + dc
                if 0 <= rr < label.shape[0] and 0 <= cc < label.shape[1]:
                    label[rr, cc] = 255

    label_img = Image.fromarray(label[::-1, :])
    label_fname = f"{lon:.5f}_{lat:.5f}_{date_target_str}_label.png"
    label_path = os.path.join(out_dir, label_fname)
    label_img.save(label_path)


def process_all_tiles(src, out_dir, step, date_target_str=None):
    # Get raster bounds in WGS84
    bounds = src.bounds
    wgs84_crs = 'EPSG:4326'
    src_crs = src.crs

    # Reproject raster bounds into WGS84 (in case raster CRS != EPSG:4326)
    try:
        from rasterio.warp import transform_bounds
        min_lon, min_lat, max_lon, max_lat = transform_bounds(src_crs, wgs84_crs,
                                                              bounds.left, bounds.bottom,
                                                              bounds.right, bounds.top,
                                                              densify_pts=21)
    except Exception as e:
        print(f"Warning: could not transform bounds: {e}")
        return

    # Figure out how many tiles total (for progress bar)
    n_lon = int(math.ceil((max_lon - min_lon) / step))
    n_lat = int(math.ceil((max_lat - min_lat) / step))
    total_tiles = n_lon * n_lat

    with tqdm(total=total_tiles, desc=f"Processing tiles ({date_target_str or 'eval'})", unit="tile") as pbar:
        lon = min_lon
        while lon < max_lon:
            lat = min_lat
            while lat < max_lat:
                try:
                    # Transform tile window to raster CRS
                    x1, y1, x2, y2 = transform_bounds(
                        wgs84_crs, src_crs, lon, lat, lon + step, lat + step
                    )
                    # Convert to pixel indices
                    row1, col1 = src.index(x1, y1)
                    row2, col2 = src.index(x2, y2)
                    row_start, row_end = sorted([max(0, row1), min(src.height, row2)])
                    col_start, col_end = sorted([max(0, col1), min(src.width, col2)])

                    if row_start < row_end and col_start < col_end:
                        data = src.read([1, 2, 3], window=((row_start, row_end), (col_start, col_end)))
                        if data.size > 0 and not (np.all(np.isnan(data)) or np.all(data == 0)):
                            r, g, b = data.astype(float)
                            grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
                            save_greyscale_only(grey, lon, lat, step, out_dir, date_target_str)
                except Exception:
                    pass
                lat += step
                pbar.update(1)
            lon += step


def group_coords(features, step=0.001):
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


def process_group(src, feats, lon, lat, step, out_dir, src_crs, wgs84_crs, date_target_str=None):
    try:
        # Transform bounding box from WGS84 (lon/lat) to raster CRS
        x1, y1, x2, y2 = transform_bounds(
            wgs84_crs, src_crs, lon - step, lat - step, lon + 2 * step, lat + 2 * step
        )
        # Convert world coordinates to pixel indices
        row1, col1 = src.index(x1, y1)
        row2, col2 = src.index(x2, y2)
        row_start, row_end = sorted([row1, row2])
        col_start, col_end = sorted([col1, col2])

        # Read RGB bands and convert to greyscale
        data = src.read([1, 2, 3], window=((row_start, row_end), (col_start, col_end)))
        if data.size == 0 or np.all(np.isnan(data)) or np.all(data == 0):
            value = None
        else:
            r = data[0].astype(float)
            g = data[1].astype(float)
            b = data[2].astype(float)
            grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
            value = grey
            save_greyscale_and_label(grey, feats, lon, lat, step, out_dir, date_target_str)
    except Exception:
        value = None

    if value is not None:
        print(f"Region ({lon:.6f}, {lat:.6f}) to ({lon+step:.6f}, {lat+step:.6f}): Greyscale value: {value.shape}, Features: {len(feats)}")


def parse_date_safe(date_str):
    """Parse YYYY-MM-DD or return None if missing/invalid."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def filter_tents_by_target_date(features, date_target):
    """Keep tents where start <= target <= end, or end is None."""
    filtered = []
    for feat in features:
        props = feat.get('properties', {})
        tent_start_dt = parse_date_safe(props.get('date_start'))
        tent_end_dt = parse_date_safe(props.get('date_end'))
        if tent_start_dt and tent_start_dt <= date_target:
            if tent_end_dt and date_target <= tent_end_dt:
                filtered.append(feat)
            elif tent_end_dt is None and date_target - tent_start_dt <= timedelta(days=30):
                filtered.append(feat)
    return filtered


def extract_date_from_filename(filename):
    """Extract first numeric sequence (YYYYMMDD) from a filename split by underscores."""
    parts = os.path.splitext(os.path.basename(filename))[0].split("_")
    for p in parts:
        if p.isdigit() and len(p) == 8:  # likely a date
            return p
    return None


def is_high_quality_tile(feats, date_target_str, src, lon, lat, step=0.001, start_threshold=0.2, max_missing_end=0.6, min_valid_fraction=0.9):
    """
    Determine if a tile is high quality based on:
    1. Fraction of tents starting on the target date
    2. Fraction of tents with missing end date
    3. Completeness of raster data in the tile
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
        # Transform WGS84 bounding box to raster CRS
        x1, y1, x2, y2 = transform_bounds(
            'EPSG:4326', src.crs, lon - step, lat - step, lon + 2 * step, lat + 2 * step
        )
        # Get pixel indices, clipped to raster bounds
        row1, col1 = src.index(x1, y1)
        row2, col2 = src.index(x2, y2)
        row_start, row_end = max(0, min(row1, row2)), min(src.height, max(row1, row2))
        col_start, col_end = max(0, min(col1, col2)), min(src.width, max(col1, col2))
        if row_start >= row_end or col_start >= col_end:
            return False
        # Empty window
        data = src.read([1, 2, 3], window=((row_start, row_end), (col_start, col_end)))
        # Fraction of valid (non-NaN, non-zero) pixels
        nodata = src.nodata
        if nodata is not None:
            valid_mask = data != nodata
        else:
            valid_mask = data != 0
        valid_fraction = np.count_nonzero(valid_mask) / data.size
        if valid_fraction < min_valid_fraction:
            return False
    except Exception:
        return False

    return True


def scan_grouped_coordinates(geotiff_path, geojson_path, out_dir, step=0.001, date_target_str=None):
    valid_tile_count = 0  # Track valid tiles for this GeoTIFF

    try:
        src = rasterio.open(geotiff_path)
    except RasterioIOError as e:
        print(f"Error opening GeoTIFF: {e}")
        return 0  # Return 0 if file can't be opened

    with open(geojson_path, 'r') as f:
        geojson = json.load(f)
    features = geojson.get('features', [])

    if date_target_str:
        try:
            date_obj = datetime.strptime(date_target_str, '%Y%m%d').date()
            features = filter_tents_by_target_date(features, date_obj)
            print(f"[{os.path.basename(geotiff_path)}] Filtered to {len(features)} features for date {date_target_str}.")
        except Exception as e:
            print(f"Date parsing error for {geotiff_path}: {e}")

    grouped = group_coords(features, step)
    print(f"[{os.path.basename(geotiff_path)}] Found {len(grouped)} coordinate groups with tents.")
    bounds = src.bounds
    print(f"GeoTIFF bounds: min_lon={bounds.left}, min_lat={bounds.bottom}, max_lon={bounds.right}, max_lat={bounds.top}")

    src_crs = src.crs
    wgs84_crs = 'EPSG:4326'

    for (lon, lat), feats in grouped.items():
        if not is_high_quality_tile(feats, date_target_str, src, lon, lat, step):
            continue  # skip low-quality tile
        process_group(src, feats, lon, lat, step, out_dir, src_crs, wgs84_crs, date_target_str)
        valid_tile_count += 1  # Increment for each valid tile

    if valid_tile_count == 0:
        print(f"Warning: No valid high-quality tiles found in {os.path.basename(geotiff_path)}")

    src.close()
    return valid_tile_count  # Return count of valid tiles


@click.command()
@click.option('--geotiff_dir', required=True, type=click.Path(exists=True, file_okay=False), help='Path to folder containing GeoTIFF files')
@click.option('--geojson', required=False, type=click.Path(exists=True), help='Path to the tent GeoJSON file (training mode only)')
@click.option('--output', required=True, type=click.Path(), help='Output folder for images')
@click.option('--step', default=0.001, show_default=True, type=float, help='Step size for grouping coordinates')
def main(geotiff_dir, geojson, output, step):
    tif_files = glob.glob(os.path.join(geotiff_dir, "*.tif"))
    if not tif_files:
        print(f"No .tif files found in {geotiff_dir}")
        return

    # If geojson is provided -> training mode
    if geojson:
        from datetime import datetime
        summary = {}
        for tif_path in tif_files:
            date_target_str = extract_date_from_filename(tif_path)
            if not date_target_str:
                print(f"Skipping {tif_path} (no date found in filename).")
                continue
            print(f"Processing {tif_path} with date {date_target_str}...")
            valid_count = scan_grouped_coordinates(tif_path, geojson, output, step, date_target_str)
            summary[os.path.basename(tif_path)] = valid_count

        print("\n=== Summary of Valid Tile Exports ===")
        for tif_name, count in summary.items():
            print(f"{tif_name}: {count} valid tiles")

    # If geojson not provided -> evaluation mode
    else:
        for tif_path in tif_files:
            date_target_str = extract_date_from_filename(tif_path)
            print(f"[EVAL MODE] Processing {tif_path}...")
            try:
                with rasterio.open(tif_path) as src:
                    process_all_tiles(src, output, step, date_target_str)
            except RasterioIOError as e:
                print(f"Error opening {tif_path}: {e}")


if __name__ == "__main__":
    main()

# EXAMPLE CLI USAGE (Training):
# python scrape_fa/coordinate_scanner.py \
#   --geotiff_dir data/planet_data \
#   --geojson data/historic_tents.geojson \
#   --output output/tiles \
#   --step 0.001
#
# EXAMPLE CLI USAGE (Evaluation):
# python scrape_fa/coordinate_scanner.py \
#   --geotiff_dir data/planet_data_evaluate \
#   --output output/tiles_evaluate \
#   --step 0.001
