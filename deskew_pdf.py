# Deskew and crop PDF pages
import sys
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from deskew_image import deskew_image, crop_image

def process_pdf(input_pdf, output_pdf, dpi=300):
    try:
        pages = convert_from_path(input_pdf, dpi=dpi)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return
    if not pages:
        print("Error: No pages found in PDF.")
        return
    processed_images = []
    for page in pages:
        image = np.array(page)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        deskewed, _ = deskew_image(image)
        cropped, _ = crop_image(deskewed)
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        processed_images.append(pil_img)
    processed_images[0].save(output_pdf, save_all=True, append_images=processed_images[1:])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deskew_pdf.py input.pdf")
        sys.exit(1)
    input_pdf = sys.argv[1]
    base, ext = os.path.splitext(input_pdf)
    output_pdf = f"{base}_processed{ext}"
    process_pdf(input_pdf, output_pdf)
