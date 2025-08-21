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

Ensure you have Python 3.10+ and install the required dependencies. Ideally, you will use poetry for this:

```bash
poetry install
```

Alternatively, you can manually install the list of dependencies listed in pyproject.toml with pip:

```bash
pip install -r requirements.txt
````

### Updating requirements.txt

The requirements.txt is not updated automatically, and must be regenerated with

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

on a regular basis.


### Command-Line Interface

From the root of your project, run:

**Environment variables required:**
- `GOOGLE_API_KEY` and `GDRIVE_ID` must be set (see .env file).

---
or
  max_missing_end: 0.2
  min_valid_fraction: 0.9
python -m scrape_fa.coordinate_scanner config.yaml

---
#### Configuration File (config.yaml)

The CLI requires a YAML configuration file with the following structure:

### 3. Training the CNN

Run the train-cnn script to train a model on the HDF5 dataset:

```bash
poetry run train-cnn config.yaml
```

**config.yaml example:**
```yaml
hdf5: processed_data.h5
training:
  training_frac: 0.8
  validation_frac: 0.1
  batch_size: 16
  epochs: 10
  learning_rate: 0.001
  checkpoint: null  # or path/to/checkpoint.pth
```

---

## Output

- **GeoTIFFs:** Downloaded to the directory specified in `geotiff_dir`.
- **HDF5 Dataset:** Contains two groups: `features` (greyscale images) and `labels` (masks). Each entry is a dataset with attributes: `origin_image`, `origin_date`, and `geo_bounds`.
- **Model Checkpoints & Logs:** Saved during training (see `runs/` and log files).
- **Predictions:** If generated, are saved as hdf5 files in the `runs/` directory, same structure as HDF5 ds.

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
