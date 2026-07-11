import cv2
import numpy as np


def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB.

    Uses OpenCV's cv2.PSNR internally.
    Matches the logic from filterAnalysis.py.
    """
    return cv2.PSNR(original, processed)


def calculate_snr(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio (SNR) in dB.

    SNR = 10 * log10(signal_power / noise_power)

    where:
        signal_power = mean(original ** 2)
        noise_power  = mean((original - processed) ** 2)

    Matches the logic from filterAnalysis.py exactly.
    """
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - processed) ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)