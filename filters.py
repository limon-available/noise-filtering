import cv2
import numpy as np


def gaussian_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur filter.

    Uses cv2.GaussianBlur with (kernel_size, kernel_size) kernel and sigmaX=0.
    Matches the logic from filterAnalysis.py.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Median filter.

    Uses cv2.medianBlur with given kernel_size.
    Matches the logic from filterAnalysis.py.
    """
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """Apply Bilateral filter.

    Uses cv2.bilateralFilter with given parameters.
    Matches the logic from filterAnalysis.py.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)