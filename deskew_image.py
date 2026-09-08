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


def text_projection_angle(gray):
    """Estimate small text-line skew; return zero when row evidence is weak."""
    scale = min(1.0, 1000 / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    valid = [i for i, (x, y, width, height, area) in enumerate(stats[1:], 1)
             if 3 <= height <= h * 0.1 and 1 <= width <= w * 0.12
             and 4 <= area <= h * w * 0.005 and width <= height * 6]
    if len(valid) < 20:
        return 0.0
    lookup = np.zeros(len(stats), np.uint8)
    lookup[valid] = 1
    mask = lookup[labels]
    if mask.mean() > 0.35:
        return 0.0
    # Padding prevents angle-dependent clipping from influencing the score.
    pad = math.ceil(max(h, w) * 0.3)
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    mh, mw = mask.shape

    def projection(angle):
        matrix = cv2.getRotationMatrix2D((mw / 2, mh / 2), float(angle), 1)
        rotated = cv2.warpAffine(mask, matrix, (mw, mh), flags=cv2.INTER_NEAREST)
        return rotated.sum(axis=1, dtype=np.float64)

    def score(angle):
        rows = projection(angle)
        return float(np.dot(rows, rows))

    coarse = np.arange(-15.0, 15.01, 1.0)
    scores = np.array([score(a) for a in coarse])
    best = float(coarse[int(np.argmax(scores))])
    fine = np.arange(max(-15, best - 1), min(15, best + 1) + 0.01, 0.25)
    fine_scores = np.array([score(a) for a in fine])
    angle = float(fine[int(np.argmax(fine_scores))])
    peak = float(np.max(fine_scores))
    distant = scores[np.abs(coarse - angle) >= 2]
    if abs(angle) >= 14.75 or abs(angle) < 0.25 or peak < score(0) * 1.08:
        return 0.0
    if distant.size and peak < float(distant.max()) * 1.1:
        return 0.0
    rows = projection(angle)
    active = (rows > rows.max() * 0.2).astype(np.int8)
    starts = np.flatnonzero(np.diff(np.pad(active, (1, 1))) == 1)
    ends = np.flatnonzero(np.diff(np.pad(active, (1, 1))) == -1)
    if np.count_nonzero(ends - starts >= 3) < 3:
        return 0.0
    return angle


def deskew_image(img):
    validate_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1600 / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, max(60, int(min(gray.shape) * 0.2)))
    angle = None
    if lines is not None:
        angles = (np.degrees(lines[:, 0, 1]) - 90 + 45) % 90 - 45
        candidates = angles[np.abs(angles) <= 15]
        if len(candidates) >= 2 and len(candidates) >= len(angles) * 0.75:
            candidate = float(np.median(candidates))
            if np.mean(np.abs(candidates - candidate) <= 2) >= 0.75:
                angle = candidate
    if angle is None:
        angle = text_projection_angle(gray)
    if abs(angle) < 0.25:
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
