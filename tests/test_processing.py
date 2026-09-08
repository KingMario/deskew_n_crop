"""Regression checks for scan correctness and failure reporting."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import fitz as pymupdf
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import deskew_image as images
import deskew_pdf as pdfs


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scan-regression-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def blank(self):
        return np.full((1000, 800, 3), 255, np.uint8)

    def test_upright_horizontal_vertical_and_rectangle(self):
        for vertical, horizontal in ((True, False), (False, True), (True, True)):
            with self.subTest(vertical=vertical, horizontal=horizontal):
                img = self.blank()
                if vertical:
                    for x in (100, 700):
                        cv2.line(img, (x, 100), (x, 900), (0, 0, 0), 3)
                if horizontal:
                    for y in (100, 900):
                        cv2.line(img, (100, y), (700, y), (0, 0, 0), 3)
                result, angle = images.deskew_image(img)
                self.assertEqual(angle, 0)
                np.testing.assert_array_equal(result, img)

    def test_actual_tilt_is_corrected(self):
        img = self.blank()
        for y in range(200, 801, 100):
            cv2.line(img, (150, y), (650, y), (0, 0, 0), 3)
        matrix = cv2.getRotationMatrix2D((400, 500), 8, 1)
        tilted = cv2.warpAffine(img, matrix, (800, 1000), borderValue=(255, 255, 255))
        corrected, angle = images.deskew_image(tilted)
        self.assertAlmostEqual(angle, -8, delta=1)
        _, residual = images.deskew_image(corrected)
        self.assertAlmostEqual(residual, 0, delta=1)

    def test_crop_retains_illustration_and_text(self):
        img = self.blank()
        cv2.rectangle(img, (100, 100), (500, 300), (0, 0, 0), -1)
        cv2.putText(img, "Example text", (150, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        result, box = images.crop_image(img)
        self.assertLessEqual(box[0], 100)
        self.assertLessEqual(box[1], 100)
        self.assertGreater(box[1] + box[3], 700)
        self.assertEqual(np.count_nonzero(np.min(img, axis=2) < 250),
                         np.count_nonzero(np.min(result, axis=2) < 250))

    def test_rotated_corner_markers_survive(self):
        img = self.blank()
        for x, y in ((0, 0), (769, 0), (0, 969), (769, 969)):
            cv2.rectangle(img, (x, y), (x + 30, y + 30), (0, 0, 255), -1)
        lines = np.array([[[0, np.deg2rad(100)]], [[10, np.deg2rad(100)]]], np.float32)
        with patch.object(images.cv2, "HoughLines", return_value=lines):
            result, _ = images.deskew_image(img)
        red = ((result[:, :, 2] > 200) & (result[:, :, 1] < 50)).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(red)
        self.assertEqual(count - 1, 4)
        self.assertTrue(all(stats[1:, cv2.CC_STAT_AREA] > 850))

    def test_blank_retained(self):
        img = self.blank()
        result, angle = images.deskew_image(img)
        cropped, box = images.crop_image(result)
        self.assertEqual(angle, 0)
        self.assertIsNone(box)
        np.testing.assert_array_equal(img, cropped)

    def test_failed_image_write_raises_and_leaves_no_output(self):
        target = self.root / "output.png"
        with patch.object(images.cv2, "imwrite", return_value=False):
            with self.assertRaises(OSError):
                images.write_image(target, self.blank())
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_inputs_exit_nonzero(self):
        for name in ("deskew_image.py", "deskew_pdf.py"):
            result = subprocess.run([sys.executable, "-B", str(SCRIPTS / name),
                                     str(self.root / "missing")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Error:", result.stderr)

    def test_existing_output_preserved(self):
        target = self.root / "existing.png"
        target.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            images.write_image(target, self.blank())
        self.assertEqual(target.read_bytes(), b"keep me")

    def make_pdf(self, pages=3):
        source = self.root / "input.pdf"
        with pymupdf.open() as document:
            for _ in range(pages):
                document.new_page(width=595.2, height=841.92)
            document.save(source)
        return source

    def test_pdf_dpi_size_and_multi_page_output(self):
        source = self.make_pdf()
        target = self.root / "output.pdf"
        with contextlib.redirect_stdout(io.StringIO()):
            pdfs.process_pdf(source, target, dpi=72)
        with pymupdf.open(target) as document:
            self.assertEqual(document.page_count, 3)
            for page in document:
                self.assertAlmostEqual(page.rect.width, 595.2, delta=1)
                self.assertAlmostEqual(page.rect.height, 841.92, delta=1)
        # Physical page dimensions are independent of rendering DPI.
        target2 = self.root / "output300.pdf"
        with contextlib.redirect_stdout(io.StringIO()):
            pdfs.process_pdf(source, target2, dpi=300)
        with pymupdf.open(target2) as document:
            self.assertAlmostEqual(document[0].rect.width, 595.2, delta=0.3)
            self.assertAlmostEqual(document[0].rect.height, 841.92, delta=0.3)

    def test_mid_pdf_failure_preserves_existing_output(self):
        source = self.make_pdf()
        target = self.root / "output.pdf"
        target.write_bytes(b"existing")
        original = pdfs.deskew_image
        calls = 0

        def fail_second(img):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated page failure")
            return original(img)

        with patch.object(pdfs, "deskew_image", side_effect=fail_second):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(OSError):
                pdfs.process_pdf(source, target, dpi=72, force=True)
        self.assertEqual(target.read_bytes(), b"existing")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["input.pdf", "output.pdf"])

    def test_pdf_input_overwrite_rejected(self):
        source = self.make_pdf()
        original = source.read_bytes()
        with self.assertRaises(ValueError):
            pdfs.process_pdf(source, source, force=True)
        self.assertEqual(source.read_bytes(), original)

    def test_faint_marks_retained(self):
        img = self.blank()
        img[5:10, 5:10] = 254
        img[500:510, 500:510] = 0
        cropped, box = images.crop_image(img)
        self.assertEqual(box[:2], (0, 0))
        self.assertEqual(np.count_nonzero(np.min(cropped, axis=2) < 255), 125)

    def test_transparency_composited_onto_white(self):
        source = self.root / "transparent.png"
        img = np.zeros((100, 100, 4), np.uint8)
        img[40:60, 40:60, 3] = 255
        cv2.imwrite(str(source), img)
        with contextlib.redirect_stdout(io.StringIO()):
            corrected, cropped, _, _ = images.process_image_file(source)
        result = cv2.imread(corrected)
        self.assertTrue(np.all(result[0, 0] == 255))
        self.assertTrue(np.all(result[50, 50] == 0))
        self.assertEqual(cv2.imread(cropped).shape[:2], (44, 44))

    def test_multiframe_and_high_bit_depth_rejected(self):
        multi = self.root / "multi.tiff"
        cv2.imwritemulti(str(multi), [self.blank(), self.blank()])
        with self.assertRaisesRegex(ValueError, "Multi-frame"):
            images.process_image_file(multi)
        deep = self.root / "deep.png"
        cv2.imwrite(str(deep), np.full((50, 50), 65535, np.uint16))
        with self.assertRaisesRegex(ValueError, "Unsupported image mode"):
            images.process_image_file(deep)
        self.assertFalse(list(self.root.glob("*_corrected.*")))

    def test_exif_orientation(self):
        source = self.root / "oriented.jpg"
        img = Image.new("RGB", (80, 40), "white")
        exif = img.getexif()
        exif[274] = 6
        img.save(source, exif=exif)
        with contextlib.redirect_stdout(io.StringIO()):
            corrected, _, _, _ = images.process_image_file(source)
        self.assertEqual(cv2.imread(corrected).shape[:2], (80, 40))

    def test_oversize_rejected_before_decode(self):
        source = self.root / "large.png"
        Image.new("RGB", (100, 100), "white").save(source)
        with patch.object(images, "MAX_PIXELS", 100), patch.object(Image.Image, "load", side_effect=AssertionError("decoded")):
            with self.assertRaisesRegex(ValueError, "pixel limit"):
                images.process_image_file(source)

    def test_pdf_nonempty_page_content_and_order(self):
        source = self.root / "colors.pdf"
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        with pymupdf.open() as doc:
            for color in colors:
                page = doc.new_page(width=200, height=300)
                page.draw_rect(pymupdf.Rect(50, 80, 150, 220), color=color, fill=color)
            doc.save(source)
        target = self.root / "colors_processed.pdf"
        with contextlib.redirect_stdout(io.StringIO()):
            pdfs.process_pdf(source, target, dpi=72)
        with pymupdf.open(target) as doc:
            self.assertEqual(doc.page_count, 3)
            for page, expected in zip(doc, colors):
                pix = page.get_pixmap(colorspace=pymupdf.csRGB, alpha=False)
                pixels = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
                np.testing.assert_array_equal(pixels[pix.height // 2, pix.width // 2], np.array(expected) * 255)

    def test_non_pdf_rejected(self):
        source = self.root / "image.png"
        Image.new("RGB", (20, 20), "white").save(source)
        target = self.root / "output.pdf"
        with self.assertRaisesRegex(ValueError, "must be a PDF"):
            pdfs.process_pdf(source, target)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
