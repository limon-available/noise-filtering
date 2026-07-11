import numpy as np

from filters import gaussian_filter, median_filter, bilateral_filter
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
        One of "gaussian", "median", or "bilateral".
    iterations : int
        Number of filtering iterations (default 5).

    Returns
    -------
    images : list[np.ndarray]
        List of [noisy, filtered_1, filtered_2, ..., filtered_N].
        Each element is the image after that many filter applications.
    psnr_values : list[float]
        PSNR (dB) of each filtered image compared to the original.
        Length = iterations (one value per iteration).
    snr_values : list[float]
        SNR (dB) of each filtered image compared to the original.
        Length = iterations (one value per iteration).
    """
    images = [noisy]
    psnr_values = []
    snr_values = []

    current = noisy.copy()

    for i in range(iterations):
        if filter_name == "gaussian":
            current = gaussian_filter(current)
        elif filter_name == "median":
            current = median_filter(current)
        elif filter_name == "bilateral":
            current = bilateral_filter(current)
        else:
            raise ValueError(f"Unknown filter: '{filter_name}'. Use 'gaussian', 'median', or 'bilateral'.")

        images.append(current.copy())

        psnr = calculate_psnr(original, current)
        snr = calculate_snr(original, current)

        psnr_values.append(psnr)
        snr_values.append(snr)

    return images, psnr_values, snr_values