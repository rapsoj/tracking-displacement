"""
Green Dashed Line Detection Script

This script processes screenshots to detect areas outlined with green dashed lines
and creates binary masks where detected regions are marked as 1 and others as 0.
"""

import cv2
import numpy as np
from PIL import Image
import argparse
import os
from typing import Tuple, List, Optional


class ImageProcessor:
    def __init__(self,
                 crop_top: int = 90,
                 crop_bottom: int = 90,
                 crop_left: int = 330,
                 crop_right: int = 0,
                 green_lower: Tuple[int, int, int] = (40, 120, 40),
                 green_upper: Tuple[int, int, int] = (190, 255, 190),
                 dash_min_length: int = 2,
                 dash_max_gap: int = 20):
        """
        Initialize the Green Dashed Line Detector

        Args:
            crop_top: Pixels to crop from top
            crop_bottom: Pixels to crop from bottom
            crop_left: Pixels to crop from left
            crop_right: Pixels to crop from right
            green_lower: Lower RGB bound for green color detection (B, G, R)
            green_upper: Upper RGB bound for green color detection (B, G, R)
            dash_min_length: Minimum length of dash segments
            dash_max_gap: Maximum gap between dashes to consider them connected
        """
        self.crop_settings = (crop_top, crop_bottom, crop_left, crop_right)
        self.green_lower = np.array(green_lower)
        self.green_upper = np.array(green_upper)
        self.dash_min_length = dash_min_length
        self.dash_max_gap = dash_max_gap

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """Crop the image according to the specified settings"""
        crop_top, crop_bottom, crop_left, crop_right = self.crop_settings
        h, w = image.shape[:2]

        # Calculate crop boundaries
        top = crop_top
        bottom = h - crop_bottom if crop_bottom > 0 else h
        left = crop_left
        right = w - crop_right if crop_right > 0 else w

        return image[top:bottom, left:right]

    def detect_green_regions(self, image: np.ndarray) -> np.ndarray:
        """Detect green colored regions in the image using RGB color space"""
        # Use RGB color space directly (OpenCV loads as BGR)
        # Create mask for green colors in BGR format
        green_mask = cv2.inRange(image, self.green_lower, self.green_upper)
        print(green_mask)

        return green_mask

    def detect_dashed_lines(self, green_mask: np.ndarray) -> List[np.ndarray]:
        """Detect dashed line patterns in the green mask"""
        # Find contours in the green mask
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dashed_line_contours = []

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate contour area and perimeter for additional filtering
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            dashed_line_contours.append(contour)

        return dashed_line_contours

    def find_enclosed_regions(self, image: np.ndarray, dashed_contours: List[np.ndarray]) -> np.ndarray:
        """Find regions that are enclosed by dashed lines or completed by image edges"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if not dashed_contours:
            return mask

        # Create a temporary image with dashed lines drawn
        line_image = np.zeros((h, w), dtype=np.uint8)

        # Draw all dashed line contours
        for contour in dashed_contours:
            cv2.drawContours(line_image, [contour], -1, 255, thickness=2)

        # Apply morphological operations to connect nearby dashes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dash_max_gap, self.dash_max_gap))
        connected_lines = cv2.morphologyEx(line_image, cv2.MORPH_CLOSE, kernel)

        # Create extended image by mirroring to each edge
        # Mirror horizontally and vertically to create a 3x3 grid with original in center
        extended_image = np.zeros((3 * h, 3 * w), dtype=np.uint8)

        # Fill the 3x3 grid with mirrored versions
        # Top row
        extended_image[0:h, 0:w] = cv2.flip(cv2.flip(connected_lines, 0), 1)  # flip both axes
        extended_image[0:h, w:2*w] = cv2.flip(connected_lines, 0)  # flip vertically
        extended_image[0:h, 2*w:3*w] = cv2.flip(cv2.flip(connected_lines, 0), 1)  # flip both axes

        # Middle row
        extended_image[h:2*h, 0:w] = cv2.flip(connected_lines, 1)  # flip horizontally
        extended_image[h:2*h, w:2*w] = connected_lines  # original image
        extended_image[h:2*h, 2*w:3*w] = cv2.flip(connected_lines, 1)  # flip horizontally

        # Bottom row
        extended_image[2*h:3*h, 0:w] = cv2.flip(cv2.flip(connected_lines, 0), 1)  # flip both axes
        extended_image[2*h:3*h, w:2*w] = cv2.flip(connected_lines, 0)  # flip vertically
        extended_image[2*h:3*h, 2*w:3*w] = cv2.flip(cv2.flip(connected_lines, 0), 1)  # flip both axes

        # Find contours in the extended image
        contours, _ = cv2.findContours(extended_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask for the extended image
        extended_mask = np.zeros((3 * h, 3 * w), dtype=np.uint8)

        # Fill enclosed areas
        for contour in contours:
            # Check if the contour forms a reasonable enclosed shape
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                cv2.fillPoly(extended_mask, [contour], 1)

        # Extract only the original image region (center of the 3x3 grid)
        mask = extended_mask[h:2*h, w:2*w]

        return mask

    def process_label(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Process a single image to detect green dashed line regions

        Args:
            image_path: Path to input image
            output_path: Optional path to save the mask image

        Returns:
            Binary mask array where 1 indicates areas within green dashed lines
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Crop image
        cropped_image = self.crop_image(image)

        # Detect green regions
        green_mask = self.detect_green_regions(cropped_image)

        # Detect dashed line patterns
        dashed_contours = self.detect_dashed_lines(green_mask)

        # Find enclosed regions
        final_mask = self.find_enclosed_regions(cropped_image, dashed_contours)

        # Save mask if output path provided
        if output_path:
            # Convert binary mask to 0-255 range for saving
            mask_image = (final_mask * 255).astype(np.uint8)
            self.save_image(mask_image, output_path)
        return final_mask

    def save_image(self, image: np.ndarray, output_path: str):
        cv2.imwrite(output_path, image)

    def process_feature(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Process a single image to crop to same dimensions

        Args:
            image_path: Path to input image
            output_path: Optional path to save the mask image

        Returns:
            Binary mask array where 1 indicates areas within green dashed lines
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        cropped_img = self.crop_image(img)
        if output_path:
            cv2.imwrite(output_path, cropped_img)
        return cropped_img


    def process_directory(self, input_dir: str, output_dir: str, pattern: str = "screenshot_*.png") -> List[Tuple[int, int]]:
        """
        Process all images in a directory matching the given pattern

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save mask images
            pattern: File pattern to match (supports glob)

        Returns:
            List of (i, j) pairs that were saved because they had white pixels
        """
        import glob
        import re

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Find all matching files
        search_pattern = os.path.join(input_dir, pattern)
        image_files = glob.glob(search_pattern)

        print(f"Found {len(image_files)} images to process")

        saved_pairs = []

        for image_path in image_files:
            try:
                # Generate output filename
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_mask{ext}"
                output_path = os.path.join(output_dir, output_filename)

                # Process image
                mask = self.process_label(image_path)

                # Print statistics
                total_pixels = mask.size
                marked_pixels = np.sum(mask)
                percentage = (marked_pixels / total_pixels) * 100

                # Only save if there are white pixels (marked pixels > 0)
                if marked_pixels > 0:
                    # Convert binary mask to 0-255 range for saving
                    mask_image = (mask * 255).astype(np.uint8)
                    self.save_image(mask_image, output_path)

                    # Extract i, j from filename using regex
                    match = re.search(r'i_(\d+)_j_(\d+)', filename)
                    if match:
                        i, j = int(match.group(1)), int(match.group(2))
                        saved_pairs.append((i, j))

                    print(f"Processed {filename}: {marked_pixels}/{total_pixels} pixels marked ({percentage:.2f}%) - SAVED")
                else:
                    print(f"Processed {filename}: {marked_pixels}/{total_pixels} pixels marked ({percentage:.2f}%) - SKIPPED (no white pixels)")

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

        print(f"Saved {len(saved_pairs)} files with white pixels")
        return saved_pairs

    def crop_directory(self, input_dir: str, output_dir: str, pattern: str = "screenshot_*.png", ij_pairs: Optional[List[Tuple[int, int]]] = None):
        """
        Crop all images in a directory matching the given pattern

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save cropped images
            pattern: File pattern to match (supports glob)
            ij_pairs: Optional list of (i, j) pairs to filter which files to crop
        """
        import glob
        import re

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Find all matching files
        search_pattern = os.path.join(input_dir, pattern)
        image_files = glob.glob(search_pattern)

        print(f"Found {len(image_files)} images")

        # Filter by i,j pairs if provided
        if ij_pairs is not None:
            ij_set = set(ij_pairs)
            filtered_files = []

            for image_path in image_files:
                filename = os.path.basename(image_path)
                match = re.search(r'i_(\d+)_j_(\d+)', filename)
                if match:
                    i, j = int(match.group(1)), int(match.group(2))
                    if (i, j) in ij_set:
                        filtered_files.append(image_path)

            image_files = filtered_files
            print(f"Filtered to {len(image_files)} images matching i,j pairs")

        print(f"Processing {len(image_files)} images to crop")

        for image_path in image_files:
            try:
                # Generate output filename
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_cropped{ext}"
                output_path = os.path.join(output_dir, output_filename)

                # Crop image using process_feature method
                cropped_img = self.process_feature(image_path, output_path)

                # Print statistics
                h, w = cropped_img.shape[:2]
                print(f"Cropped {filename}: {w}x{h} pixels")

            except Exception as e:
                print(f"Error cropping {image_path}: {str(e)}")
