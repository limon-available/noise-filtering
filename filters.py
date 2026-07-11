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


def box_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Box (Averaging) filter.

    Uses cv2.blur with (kernel_size, kernel_size) kernel.
    Each output pixel is the mean of its kernel neighbors.
    """
    return cv2.blur(image, (kernel_size, kernel_size))


def weighted_smoothing_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Weighted Smoothing filter using a Gaussian-weighted kernel.

    Uses cv2.GaussianBlur with a larger sigma to produce
    smoother weighting than the standard Gaussian filter.
    """
    sigma = kernel_size / 3.0
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def sobel_filter(image: np.ndarray) -> np.ndarray:
    """Apply Sobel edge detection filter.

    Computes gradient magnitude from Sobel X and Y derivatives.
    Result is normalized to 0-255 uint8 range.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)


def laplacian_filter(image: np.ndarray) -> np.ndarray:
    """Apply Laplacian edge detection filter.

    Uses cv2.Laplacian to compute second-order derivatives.
    Result is normalized to 0-255 uint8 range.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)
    return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)


def unsharp_masking(image: np.ndarray, kernel_size: int = 5, alpha: float = 1.5) -> np.ndarray:
    """Apply Unsharp Masking filter.

    Sharpens the image by adding the difference between
    the original and a blurred version.
    result = image + alpha * (image - blurred)
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    sharpened = cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def high_boost_filter(image: np.ndarray, kernel_size: int = 5, boost: float = 2.0) -> np.ndarray:
    """Apply High Boost Filtering.

    Enhances high-frequency components.
    result = boost * image - (boost - 1) * blurred
    Equivalent to: image + (boost - 1) * (image - blurred)
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    result = cv2.addWeighted(image, boost, blurred, 1.0 - boost, 0)
    return np.clip(result, 0, 255).astype(np.uint8)