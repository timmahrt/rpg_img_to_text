from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
import cv2

# import numpy as np
# import os


def preprocess(img):
    # img = np.array(image)

    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.0)  # Adjust the enhancement factor as needed

    img = img.convert("L")  # Convert to grayscale
    threshold = 128  # Adjust the threshold as needed
    img = img.point(lambda p: p > threshold and 255)

    return img


def cropImage(image):
    # TODO: automatically find crop region

    # Define the coordinates of the cropping region (left, upper, right, lower)
    left = 255  # X-coordinate of the left edge of the cropping region
    upper = 65  # Y-coordinate of the upper edge of the cropping region
    right = 1000  # X-coordinate of the right edge of the cropping region
    lower = 200  # Y-coordinate of the lower edge of the cropping region

    # Crop the image
    return image.crop((left, upper, right, lower))


def textboxDetection(image):
    # Start with a small threshold and scale
    # threshold2 is generally 3 times threshold1
    threshold1 = 75
    threshold2 = threshold1 * 3

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1, threshold2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    return cv2.rectangle(
        image, (x, y), (x + w, y + h), (0, 255, 0), 2
    )  # Green rectangle around the largest contour


def ocr(image):
    config = '--tessdata-dir "tesseract_data"'
    return (
        pytesseract.image_to_string(image, lang="jpn", config=config)
        .replace(" ", "")
        .strip()
    )
