import cv2
import numpy as np
import sys

# Get image filename from CLI argument
if len(sys.argv) < 2:
    print("Usage: python deskew-image.py <image_filename>")
    sys.exit(1)
image_filename = sys.argv[1]

# Generate output filenames based on input filename
import os
base, ext = os.path.splitext(image_filename)
corrected_filename = f"{base}_corrected{ext}"
cropped_filename = f"{base}_cropped{ext}"

img = cv2.imread(image_filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Hough transform to detect lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
angles = []
print(f"Totally {len(lines)} lines detected:")

for line in lines:
    rho, theta = line[0]
    angle = (theta * 180 / np.pi) - 90  # Convert to degrees
    angles.append(angle)
    print(f"Line {len(angles)}: rho={rho:.2f}, theta={theta:.2f} ({angle:.2f}°)")

# Calculate median angle
median_angle = np.median(angles)
print(f"Calculated tilt angle: {median_angle:.2f}°")

# Rotate image
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# Save corrected image
cv2.imwrite(corrected_filename, rotated)

# Auto crop useless margins
rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # Use MSER to detect text regions and remove all margins
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(rotated_gray)
    mask = np.zeros(rotated_gray.shape, dtype=np.uint8)
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.drawContours(mask, [hull], -1, 255, -1)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = rotated[y:y+h, x:x+w]
        cv2.imwrite(cropped_filename, cropped)
        print(f"Saved cropped image with only text as {cropped_filename}, crop area: x={x}, y={y}, w={w}, h={h}")
    else:
        print("No text region detected, not cropped.")
else:
    print("No valid content region detected, not cropped.")
print(f"Saved corrected image as {corrected_filename}")
