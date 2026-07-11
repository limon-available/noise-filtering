import numpy as np


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    """Add Gaussian noise with mean=0, sigma=50."""
    mean = 0
    sigma = 50
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image + gaussian, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.08) -> np.ndarray:
    """Add Salt & Pepper noise with given amount."""
    noisy = image.copy()

    num_salt = int(amount * image.shape[0] * image.shape[1])

    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy


def add_speckle_noise(image: np.ndarray) -> np.ndarray:
    """Add Speckle noise (multiplicative) with factor 0.4."""
    noise = np.random.randn(*image.shape)
    noisy = np.clip(image + image * noise * 0.4, 0, 255).astype(np.uint8)
    return noisy


def add_poisson_noise(image: np.ndarray) -> np.ndarray:
    """Add Poisson noise."""
    vals = 2 ** np.ceil(np.log2(len(np.unique(image))))
    noisy = np.random.poisson(image * vals) / vals
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy