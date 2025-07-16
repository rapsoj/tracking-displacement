import rasterio
import json
import numpy as np
import math
from collections import defaultdict
from rasterio.errors import RasterioIOError
import os
from PIL import Image
import click

def save_greyscale_and_label(grey, feats, lon, lat, step, out_dir):
    import numpy as np
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{lon:.3f}_{lat:.3f}_feat.png"
    out_path = os.path.join(out_dir, fname)
    img = Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8))
    img.save(out_path)
    # Create label image
    label = np.zeros_like(grey, dtype=np.uint8)
    for feat in feats:
        feat_lon, feat_lat = feat['geometry']['coordinates']
        # Linearly interpolate local coordinates in the window
        local_col = int(round((feat_lon - lon) / step * (label.shape[1] - 1)))
        local_row = int(round((feat_lat - lat) / step * (label.shape[0] - 1)))
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr = local_row + dr
                cc = local_col + dc
                if 0 <= rr < label.shape[0] and 0 <= cc < label.shape[1]:
                    label[rr, cc] = 255
    label_img = Image.fromarray(label[::-1, :])
    label_fname = f"{lon:.3f}_{lat:.3f}_label.png"
    label_path = os.path.join(out_dir, label_fname)
    label_img.save(label_path)

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
                grouped[(lon_group, lat_group)].append(feat)
    return grouped

def process_group(src, feats, lon, lat, step, out_dir, src_crs, wgs84_crs):
    from rasterio.warp import transform
    # Transform WGS84 (lon, lat) to raster CRS
    xs, ys = transform(wgs84_crs, src_crs, [lon, lon + step], [lat, lat + step])
    x1, x2 = xs[0], xs[1]
    y1, y2 = ys[0], ys[1]
    try:
        row1, col1 = src.index(x1, y1)
        row2, col2 = src.index(x2, y2)
        row_start, row_end = sorted([row1, row2])
        col_start, col_end = sorted([col1, col2])
        # Read RGB bands and convert to greyscale
        data = src.read([1, 2, 3], window=((row_start, row_end), (col_start, col_end)))
        if data.size == 0 or np.all(np.isnan(data)):
            value = None
        else:
            r = data[0].astype(float)
            g = data[1].astype(float)
            b = data[2].astype(float)
            grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
            value = grey
            save_greyscale_and_label(grey, feats, lon, lat, step, out_dir)
    except Exception:
        value = None
    if value is not None:
        print(f"Region ({lon:.6f}, {lat:.6f}) to ({lon+step:.6f}, {lat+step:.6f}): Greyscale value: {value.shape}, Features: {len(feats)}")

def scan_grouped_coordinates(geotiff_path, geojson_path, out_dir, step=0.001):
    # Load GeoTIFF
    try:
        src = rasterio.open(geotiff_path)
    except RasterioIOError as e:
        print(f"Error opening GeoTIFF: {e}")
        return

    # Load GeoJSON
    with open(geojson_path, 'r') as f:
        geojson = json.load(f)
        features = geojson.get('features', [])

    grouped = group_coords(features, step)
    print(f"Found {len(grouped)} coordinate groups with tents.")

    # Print geotiff bounds
    bounds = src.bounds
    print(f"GeoTIFF bounds: min_lon={bounds.left}, min_lat={bounds.bottom}, max_lon={bounds.right}, max_lat={bounds.top}")

    # Prepare coordinate transformer
    src_crs = src.crs
    wgs84_crs = 'EPSG:4326'

    for (lon, lat), feats in grouped.items():
        process_group(src, feats, lon, lat, step, out_dir, src_crs, wgs84_crs)
    src.close()

@click.command()
@click.option('--geotiff', required=True, type=click.Path(exists=True), help='Path to the GeoTIFF file')
@click.option('--geojson', required=True, type=click.Path(exists=True), help='Path to the tent GeoJSON file')
@click.option('--output', required=True, type=click.Path(), help='Output folder for images')
@click.option('--step', default=0.001, show_default=True, type=float, help='Step size for grouping coordinates')
def main(geotiff, geojson, output, step):
    scan_grouped_coordinates(geotiff, geojson, output, step)

if __name__ == "__main__":
    main()
