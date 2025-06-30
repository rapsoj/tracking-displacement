#!/usr/bin/env python3
"""
Example usage script for the Green Dashed Line Detector

This script shows how to use the GreenDashDetector class to process screenshots
and find regions outlined with green dashed lines.
"""

import os
import numpy as np

from scrape_fa.green_dash_detector import ImageProcessor



def batch_processing_labels():
    """Example: Process all screenshots in the directory"""
    print("\n=== Batch Processing ===")

    # Initialize detector with default settings
    detector = ImageProcessor()
    # Process all screenshots in the test_scrape_fa directory
    input_dir = "../labels"
    output_dir = "../masks"
    crop_dir = "../labels_cropped"

    if os.path.exists(input_dir):
        idcs = detector.process_directory(input_dir, output_dir, "screenshot_*.png")
        idcs = sorted(idcs)
        np.savetxt("../idcs.csv", np.array(idcs, dtype=np.int32), delimiter=",")
        detector.crop_directory(input_dir, crop_dir, "screenshot_*.png", ij_pairs=idcs)
        print(f"✓ All masks saved to {output_dir}")
    else:
        print(f"✗ Directory not found: {input_dir}")


def batch_processing_feats():
    input_dir = "../feats"
    output_dir = "../feats_cropped"
    detector = ImageProcessor()

    if os.path.exists(input_dir):
        detector.crop_directory(input_dir, output_dir, "screenshot_*.png")
        print(f"✓ All crops saved to {output_dir}")
    else:
        print(f"✗ Directory not found: {input_dir}")

if __name__ == "__main__":
    # Run examples
    batch_processing_feats()
