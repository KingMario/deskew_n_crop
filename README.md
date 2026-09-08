# scan-deskew-crop

Local tools for conservative deskewing and cropping of scanned images and PDFs.

## Usage

Python 3.11 and the packages in `requirements.txt` are required. PDF processing uses PyMuPDF and does not require Poppler.

```bash
python3.11 -m pip install -r requirements.txt
python3.11 deskew_image.py /path/to/scan.jpg --output-dir /tmp/scan-deskew-results
python3.11 deskew_pdf.py /path/to/scan.pdf --output-dir /tmp/scan-deskew-results --dpi 300
```

Source files remain intact. Existing output files require `--force` to replace. PDF results are image-only documents with page dimensions determined by cropped pixel size and DPI.

Images produce `_corrected` and `_cropped` files; PDFs produce `_processed.pdf`. Errors exit nonzero. Image outputs are published individually, so a failed invocation may leave one completed image.

## Behavior and limits

- Rotation uses horizontal/vertical line consensus within 15 degrees and expands the canvas with white borders. Insufficient evidence leaves orientation unchanged.
- Cropping includes every nonwhite pixel and adds a 12-pixel margin. Noise and shadows may retain extra borders.
- Image input applies EXIF orientation and composites transparency onto white. Multi-frame and high-bit-depth images are rejected explicitly. Output is 8-bit RGB and does not preserve image metadata; JPEG encoding is lossy.
- PDF pages are processed individually through a temporary disk PDF. Raster memory depends on the current page; output size and document metadata grow with page count.
- Rendered and rotated images are limited to 40 million pixels. PDF DPI must be between 36 and 600.
- PDF output does not retain searchable text, links, or editable annotations. Keep original documents and inspect results.
- Perspective distortion and 90-degree orientation detection are outside the tool's scope.

## Tests

```bash
python3.11 -B -m unittest discover -s tests -v
```

## License

MIT License. Copyright (c) 2025 Mario Studio.
