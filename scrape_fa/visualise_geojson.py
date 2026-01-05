import json
import os
import click
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds, transform
import matplotlib.pyplot as plt

class GeoJsonViewer:
    def __init__(self, geojson_path, tif_dir):
        self.tif_dir = tif_dir
        with open(geojson_path, 'r') as f:
            self.data = json.load(f)

        self.features = self.data['features']
        self.polygons = [f for f in self.features if f['geometry']['type'] == 'Polygon']
        self.points = [f for f in self.features if f['geometry']['type'] == 'Point']

        self.index = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        if not self.polygons:
            print("No polygons found in GeoJSON.")
            return

        self.update()
        plt.show()

    def on_key(self, event):
        if event.key == 'right':
            self.index = (self.index + 1) % len(self.polygons)
            self.update()
        elif event.key == 'left':
            self.index = (self.index - 1) % len(self.polygons)
            self.update()

    def update(self):
        self.ax.clear()
        poly = self.polygons[self.index]
        source = poly['properties'].get('source')

        if not source:
            self.ax.text(0.5, 0.5, "No source specified", ha='center')
            self.fig.canvas.draw()
            return

        tif_path = os.path.join(self.tif_dir, source)
        if not os.path.exists(tif_path):
            self.ax.text(0.5, 0.5, f"TIF not found: {source}\nPath: {tif_path}", ha='center')
            self.fig.canvas.draw()
            return

        coords = poly['geometry']['coordinates'][0]
        if not coords:
            self.ax.text(0.5, 0.5, "Empty polygon coordinates", ha='center')
            self.fig.canvas.draw()
            return

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        try:
            with rasterio.open(tif_path) as src:
                # Transform bounds to match TIF CRS
                if src.crs:
                    left, bottom, right, top = transform_bounds('EPSG:4326', src.crs, min_lon, min_lat, max_lon, max_lat)
                else:
                    left, bottom, right, top = min_lon, min_lat, max_lon, max_lat

                window = from_bounds(left, bottom, right, top, src.transform)
                img = src.read(window=window)

                if img.size == 0:
                    self.ax.text(0.5, 0.5, "Region outside TIF bounds or empty", ha='center')
                    self.fig.canvas.draw()
                    return

                window_transform = src.window_transform(window)

                if img.shape[0] == 1:
                    img_disp = img[0]
                    cmap = 'gray'
                else:
                    img_disp = img.transpose(1, 2, 0)
                    # Normalize if needed? TIFs might be uint16 or float
                    if img_disp.dtype == 'uint16':
                         img_disp = (img_disp / 256).astype('uint8')
                    cmap = None

                self.ax.imshow(img_disp, cmap=cmap)

                # Find points
                points_in_poly = []
                for pt in self.points:
                    if pt['properties'].get('source') != source:
                        continue
                    pt_lon, pt_lat = pt['geometry']['coordinates']
                    if min_lon <= pt_lon <= max_lon and min_lat <= pt_lat <= max_lat:
                        # Transform point to TIF CRS
                        if src.crs:
                            xs, ys = transform('EPSG:4326', src.crs, [pt_lon], [pt_lat])
                            t_x, t_y = xs[0], ys[0]
                        else:
                            t_x, t_y = pt_lon, pt_lat

                        # Convert to pixel coords relative to window
                        r, c = rowcol(window_transform, t_x, t_y)
                        points_in_poly.append((c, r))

                if points_in_poly:
                    xs, ys = zip(*points_in_poly)
                    self.ax.scatter(xs, ys, c='red', s=40, marker='x')

                self.ax.set_title(f"Region {self.index + 1}/{len(self.polygons)}\nSource: {source}")
                self.ax.axis('off')

        except Exception as e:
            print(f"Error loading TIF: {e}")
            self.ax.text(0.5, 0.5, f"Error loading TIF: {e}", ha='center')

        self.fig.canvas.draw()

@click.command()
@click.argument('geojson_path', type=click.Path(exists=True))
@click.option('--tif-dir', default='.', help='Directory containing TIF files')
def main(geojson_path, tif_dir):
    GeoJsonViewer(geojson_path, tif_dir)

if __name__ == '__main__':
    main()

