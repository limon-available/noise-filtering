import numpy as np

from filters import (
    gaussian_filter,
    median_filter,
    bilateral_filter,
    box_filter,
    weighted_smoothing_filter,
    sobel_filter,
    laplacian_filter,
    unsharp_masking,
    high_boost_filter,
)
from metrics import calculate_psnr, calculate_snr


def analyze_filter(
    original: np.ndarray,
    noisy: np.ndarray,
    filter_name: str,
    iterations: int = 5,
):
    """
    Apply the same filter repeatedly and return all intermediate results.

    Parameters
    ----------
    original : np.ndarray
        The original (clean) image.
    noisy : np.ndarray
        The noisy image to be filtered.
    filter_name : str
        One of: "gaussian", "median", "bilateral", "box",
                "weighted_smoothing", "sobel", "laplacian",
                "unsharp_masking", "high_boost".
    iterations : int
        Number of filtering iterations (default 5).

    Returns
    -------
    images : list[np.ndarray]
        List of [noisy, filtered_1, filtered_2, ..., filtered_N].
    psnr_values : list[float]
        PSNR (dB) of each filtered image compared to the original.
    snr_values : list[float]
        SNR (dB) of each filtered image compared to the original.
    """
    images = [noisy]
    psnr_values = []
    snr_values = []

    current = noisy.copy()

    filter_map = {
        "gaussian": gaussian_filter,
        "median": median_filter,
        "bilateral": bilateral_filter,
        "box": box_filter,
        "weighted_smoothing": weighted_smoothing_filter,
        "sobel": sobel_filter,
        "laplacian": laplacian_filter,
        "unsharp_masking": unsharp_masking,
        "high_boost": high_boost_filter,
    }

    if filter_name not in filter_map:
        valid = ", ".join(filter_map.keys())
        raise ValueError(f"Unknown filter: '{filter_name}'. Valid options: {valid}")

    filter_fn = filter_map[filter_name]

    for i in range(iterations):
        current = filter_fn(current)

        images.append(current.copy())

        psnr = calculate_psnr(original, current)
        snr = calculate_snr(original, current)

        psnr_values.append(psnr)
        snr_values.append(snr)

    return images, psnr_values, snr_values