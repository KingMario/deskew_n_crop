"""Conservative scan rotation and content bounding-box cropping."""

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_PIXELS = 40_000_000


def validate_image(img):
    if img is None or img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
        raise ValueError("Expected a nonempty BGR image")
    if img.dtype != np.uint8:
        raise ValueError("Expected an 8-bit image")
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise ValueError("Image exceeds the 40 million pixel limit")


def deskew_image(img):
    validate_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1600 / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, max(60, int(min(gray.shape) * 0.2)))
    if lines is None:
        return img, 0.0
    angles = (np.degrees(lines[:, 0, 1]) - 90 + 45) % 90 - 45
    candidates = angles[np.abs(angles) <= 15]
    if len(candidates) < 2 or len(candidates) < len(angles) * 0.75:
        return img, 0.0
    angle = float(np.median(candidates))
    if np.mean(np.abs(candidates - angle) <= 2) < 0.75 or abs(angle) < 0.25:
        return img, 0.0
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    width = math.ceil(w * cosine + h * sine)
    height = math.ceil(h * cosine + w * sine)
    if width * height > MAX_PIXELS:
        raise ValueError("Rotated canvas exceeds the 40 million pixel limit")
    matrix[0, 2] += (width - w) / 2
    matrix[1, 2] += (height - h) / 2
    rotated = cv2.warpAffine(img, matrix, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated, angle


def crop_image(img, margin=12):
    validate_image(img)
    if margin < 0:
        raise ValueError("Margin must be nonnegative")
    mask = np.min(img, axis=2) < 255
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return img, None
    x, y = max(0, int(cols[0]) - margin), max(0, int(rows[0]) - margin)
    right = min(img.shape[1], int(cols[-1]) + margin + 1)
    bottom = min(img.shape[0], int(rows[-1]) + margin + 1)
    return img[y:bottom, x:right], (x, y, right - x, bottom - y)


def publish(temp, destination, force=False):
    if force:
        os.replace(temp, destination)
    else:
        os.link(temp, destination)
        os.unlink(temp)


def write_image(destination, img, force=False):
    fd, temp = tempfile.mkstemp(prefix=".deskew-", suffix=destination.suffix, dir=destination.parent)
    os.close(fd)
    try:
        if not cv2.imwrite(temp, img):
            raise OSError(f"Unable to write image: {destination}")
        publish(temp, destination, force)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def output_paths(source, output_dir, suffixes, force):
    directory = Path(output_dir).expanduser().resolve() if output_dir else source.parent
    paths = [directory / f"{source.stem}{suffix}" for suffix in suffixes]
    for path in paths:
        if path.resolve() == source or (path.exists() and os.path.samefile(path, source)):
            raise ValueError("Output must not overwrite input")
        if path.exists() and not force:
            raise FileExistsError(f"Output already exists: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    return paths


def process_image_file(image_filename, output_dir=None, force=False):
    source = Path(image_filename).expanduser().resolve(strict=True)
    with Image.open(source) as original:
        if original.width * original.height > MAX_PIXELS:
            raise ValueError("Image exceeds the 40 million pixel limit")
        if getattr(original, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are not supported; provide individual pages or a PDF")
        if original.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "CMYK"):
            raise ValueError(f"Unsupported image mode: {original.mode}; provide an 8-bit scan")
        oriented = ImageOps.exif_transpose(original).convert("RGBA")
        background = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(background, oriented).convert("RGB")
        img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    validate_image(img)
    paths = output_paths(source, output_dir,
                         [f"_corrected{source.suffix}", f"_cropped{source.suffix}"], force)
    rotated, angle = deskew_image(img)
    cropped, box = crop_image(rotated)
    for path, result in zip(paths, (rotated, cropped)):
        write_image(path, result, force)
        print(f"Saved: {path}")
    print(f"Rotation: {angle:.2f} degrees; crop: {box}")
    return str(paths[0]), str(paths[1]), angle, box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        process_image_file(args.input, args.output_dir, args.force)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
