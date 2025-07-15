import cv2
import numpy as np
import os

def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None or len(lines) == 0:
        # No lines detected, return original
        return img, 0.0
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)
    median_angle = np.median(angles)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, median_angle

def crop_image(img):
    rotated_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(rotated_gray)
        mask = np.zeros(rotated_gray.shape, dtype=np.uint8)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(mask, [hull], -1, 255, -1)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = img[y:y+h, x:x+w]
            return cropped, (x, y, w, h)
    # If no crop possible, return original
    return img, None

def process_image_file(image_filename):
    if not os.path.isfile(image_filename):
        print(f"Error: File '{image_filename}' does not exist.")
        return None
    img = cv2.imread(image_filename)
    if img is None:
        print(f"Error: Unable to read image '{image_filename}'.")
        return None
    base, ext = os.path.splitext(image_filename)
    corrected_filename = f"{base}_corrected{ext}"
    cropped_filename = f"{base}_cropped{ext}"
    rotated, angle = deskew_image(img)
    cv2.imwrite(corrected_filename, rotated)
    cropped, crop_area = crop_image(rotated)
    if crop_area:
        cv2.imwrite(cropped_filename, cropped)
        print(f"Saved cropped image with only text as {cropped_filename}, crop area: x={crop_area[0]}, y={crop_area[1]}, w={crop_area[2]}, h={crop_area[3]}")
    else:
        print("No valid content region detected, not cropped.")
    print(f"Saved corrected image as {corrected_filename}")
    return corrected_filename, cropped_filename, angle, crop_area

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python deskew_image.py <image_filename>")
        sys.exit(1)
    image_filename = sys.argv[1]
    process_image_file(image_filename)
