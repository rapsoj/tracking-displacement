# Gaza Strip Tent Detection (TentNetFA)

This project processes high-resolution Planet satellite images of the Gaza Strip in combination with historic tent locations identified by **Forensic Architecture** — a multidisciplinary research group based at Goldsmiths, University of London, which uses architectural techniques and technologies to investigate cases of state violence and human rights violations worldwide.

---

## Overview

The goal of this work is to develop a convolutional neural network (CNN) that can predict, at the pixel level, the locations of tents in the Gaza Strip from satellite imagery. These predictions use Gaussian densities to create highly granular maps of displacement patterns over time.

This automated detection supports **Forensic Architecture's Cartography of Genocide platform**, which documents the extensive and detailed impacts of Israel’s assault on Gaza since October 2023. The platform compiles thousands of documented acts of violence and destruction affecting civilian infrastructure, homes, and critical services, as well as the use of humanitarian measures as tools of population displacement.

---

## Key Features

- **Input data:**
  - Planet satellite GeoTIFF images of the Gaza Strip.
  - GeoJSON files containing geolocated historic tent points identified by Forensic Architecture.

- **Processing:**
  - Groups tent locations by geographic coordinates into spatial windows.
  - Extracts corresponding satellite image patches and converts RGB imagery to greyscale.
  - Generates paired greyscale image patches and binary label masks marking tent locations.
  - Supports pixel-level CNN training using Gaussian density labels to predict tent presence.

- **Output:**
  - Greyscale image tiles representing satellite imagery patches.
  - Label masks indicating tent locations in the corresponding image tiles.

---

## Installation

Ensure you have Python 3.10+ and install the required dependencies:

```bash
pip install rasterio click Pillow numpy
````

Here’s the fully updated **Usage** section for your `README.md`, rewritten to match your actual code in `scrape_fa/coordinate_scanner.py`. It includes both **command-line** and **Python import** usage, reflecting the way your code is structured and how users are likely to interact with it.

---

## Usage

You can run the image+label tile extraction either via **command-line interface (CLI)** or by importing the function directly in Python.

---

### Command-Line Interface

From the root of your project, run:

```bash
python scrape_fa/coordinate_scanner.py \
  --geotiff data/gaza_image.tif \
  --geojson data/historic_tents.geojson \
  --output output/tiles \
  --step 0.001
```

### Arguments

| Argument    | Description                                                                 |
| ----------- | --------------------------------------------------------------------------- |
| `--geotiff` | Path to the input GeoTIFF satellite image file.                             |
| `--geojson` | Path to the GeoJSON file containing historic tent locations.                |
| `--output`  | Output directory for generated greyscale image patches and label masks.     |
| `--step`    | *(optional)* Step size for coordinate grouping in degrees (default: 0.001). |

---

### Programmatic Usage (Python)

You can also call the same logic directly in Python, which is useful for notebooks, pipelines, or integration into training code.

```python
from scrape_fa.coordinate_scanner import scan_grouped_coordinates

scan_grouped_coordinates(
    geotiff_path="data/gaza_image.tif",
    geojson_path="data/historic_tents.geojson",
    out_dir="output/tiles",
    step=0.001  # Optional: controls tiling granularity
)
```

---

## Output

For each spatial window (tile), two PNG files are saved in the output directory:

* `{lon}_{lat}_feat.png` — A greyscale image patch extracted from the satellite image.
* `{lon}_{lat}_label.png` — A binary mask where white pixels (255) indicate known tent locations.

These pairs are ready for use in training CNN models for pixel-level displacement detection.

---

Let me know if you'd like to follow up with a section for **how to train a model on these outputs** or how to convert this into a PyTorch `Dataset`.


## Context & Impact

This work aids Forensic Architecture’s critical investigation into the ongoing humanitarian crisis in Gaza by enabling scalable, automated detection of displacement patterns from satellite imagery. The fine-grained data helps expose the scale and methods of violence, supporting accountability and historical record-keeping.

---

## Acknowledgments

* Developed in collaboration with Forensic Architecture, Goldsmiths, University of London.
* Satellite data provided by Planet Labs.
* Inspired by Forensic Architecture’s Cartography of Genocide platform.

---

## License

The MIT License (MIT)

---

If you have any questions or want to contribute, please open an issue or submit a pull request.

---

*This project is part of a broader effort to document human rights violations with technological rigor and integrity.*
