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

Ensure you have Python 3.7+ and install the required dependencies:

```bash
pip install rasterio click Pillow numpy
````

---

## Usage

Run the processing script with:

```bash
python your_script_name.py --geotiff path/to/satellite_image.tif --geojson path/to/tents.geojson --output output_directory --step 0.001
```

### Arguments

* `--geotiff`
  Path to the GeoTIFF satellite image file.

* `--geojson`
  Path to the GeoJSON file containing historic tent point features.

* `--output`
  Directory where the output greyscale and label PNG images will be saved.

* `--step` (optional)
  Spatial grouping step size in degrees (default: 0.001).

---

## Output

For each spatial window, two PNG images are created:

* `{lon}_{lat}_feat.png`: Greyscale satellite image patch.
* `{lon}_{lat}_label.png`: Corresponding binary mask with tent locations marked.

---

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
