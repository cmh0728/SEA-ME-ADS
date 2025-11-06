import cv2
import numpy as np


def create_lane_mask(bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask highlighting likely lane pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Light CLAHE on V channel to mitigate illumination swings.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v2 = clahe.apply(v)
    hsv2 = cv2.merge([h, s, v2])
    bgr2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    sobelx = cv2.Sobel(gray_blur, cv2.CV_16S, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)

    _, binary_gray = cv2.threshold(gray_blur, 210, 255, cv2.THRESH_BINARY)
    _, sat_mask = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray_blur, 80, 160)

    white_mask = cv2.inRange(hsv2, np.array([0, 0, 180]), np.array([180, 80, 255]))

    combo = np.zeros_like(gray_blur)
    combo[
        (binary_gray == 255)
        | (sat_mask == 255)
        | (edges == 255)
        | (white_mask == 255)
    ] = 255
    return combo

