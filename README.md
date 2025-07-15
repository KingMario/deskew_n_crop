# Deskew & Crop Tools for Images and PDFs

## Overview

This project provides command-line tools to automatically deskew (correct tilt) and crop scanned images and PDF documents. It detects the main content, corrects skew, and removes unnecessary margins, leaving only the essential text or graphics.

> When scanning, the edges of the paper must be clear enough so that straight lines can be detected for tilt angle calculation.

## Features

- Deskew and crop individual image files
- Deskew and crop all pages of a PDF document
- Advanced text region detection (MSER) for precise cropping
- Outputs corrected and tightly cropped files
- Simple command-line usage

## Usage

### Image Deskew & Crop

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the image tool:

   ```bash
   python deskew_image.py <your_image_file.jpg>
   ```

   - Outputs: `<your_image_file>_corrected.jpg` and `<your_image_file>_cropped.jpg`

### PDF Deskew & Crop

1. Install dependencies (see above).

2. Run the PDF tool:

   ```bash
   python deskew_pdf.py <your_pdf_file.pdf>
   ```

   - Outputs: `<your_pdf_file>_processed.pdf` (all pages deskewed and cropped)

## License

MIT License. Copyright (c) 2025 Mario Studio.
