"""Process scanned PDF pages with bounded raster memory and explicit DPI."""

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import fitz as pymupdf

from deskew_image import MAX_PIXELS, crop_image, deskew_image, output_paths, publish


def process_pdf(input_pdf, output_pdf, dpi=300, force=False):
    if not 36 <= dpi <= 600:
        raise ValueError("DPI must be between 36 and 600")
    source = Path(input_pdf).expanduser().resolve(strict=True)
    target = Path(output_pdf).expanduser().absolute()
    if target.resolve() == source or (target.exists() and os.path.samefile(target, source)):
        raise ValueError("Output must not overwrite input")
    if target.exists() and not force:
        raise FileExistsError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(source) as document:
        if not document.is_pdf:
            raise ValueError("Input must be a PDF document")
        if document.needs_pass or not document.page_count:
            raise ValueError("PDF must be unlocked and contain pages")
        with tempfile.TemporaryDirectory(prefix=".deskew-", dir=target.parent) as directory:
            staged = Path(directory) / "result.pdf"
            for index in range(document.page_count):
                page = document[index]
                width = math.ceil(page.rect.width * dpi / 72)
                height = math.ceil(page.rect.height * dpi / 72)
                if width * height > MAX_PIXELS:
                    raise ValueError(f"Page {index + 1} exceeds the 40 million pixel limit")
                pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                img = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3),
                                   cv2.COLOR_RGB2BGR)
                del pix
                rotated, angle = deskew_image(img)
                cropped, box = crop_image(rotated)
                ok, encoded = cv2.imencode(".png", cropped)
                if not ok:
                    raise OSError(f"Unable to encode page {index + 1}")
                h, w = cropped.shape[:2]
                with pymupdf.open() as single:
                    output_page = single.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
                    output_page.insert_image(output_page.rect, stream=encoded.tobytes())
                    if index == 0:
                        single.save(staged, deflate=True)
                    else:
                        with pymupdf.open(staged) as accumulated:
                            accumulated.insert_pdf(single)
                            accumulated.saveIncr()
                del img, rotated, cropped, encoded
                print(f"Page {index + 1}/{document.page_count}: rotation={angle:.2f}; crop={box}")
            publish(staged, target, force)
    print(f"Saved: {target}")
    return str(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output-dir")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        source = Path(args.input).expanduser().resolve(strict=True)
        target, = output_paths(source, args.output_dir, ["_processed.pdf"], args.force)
        process_pdf(source, target, args.dpi, args.force)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
