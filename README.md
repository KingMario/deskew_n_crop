# Image Deskew & Crop Tool

## Overview

This tool automatically detects and corrects the tilt (skew) of scanned images, then crops out all unnecessary margins, leaving only the main content (such as text).

## Features

- Detects the main straight lines in the image to estimate the tilt angle
- Rotates the image to correct the tilt
- Uses advanced text region detection (MSER) to crop out all blank margins
- Outputs two files: a corrected image and a tightly cropped image
- Command-line interface: specify the image file to process

## Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the tool:

   ```bash
   python deskew-image.py <your_image_file.jpg>
   ```

   - The tool will output `<your_image_file>_deskewed.jpg` (deskewed) and `<your_image_file>_cropped.jpg` (cropped to text only).

## License

MIT License. Copyright (c) 2025 Mario Studio.
