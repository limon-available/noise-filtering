import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Gaussian Noise
mean = 0
sigma = 50
gaussian = np.random.normal(mean, sigma, image.shape)
gaussian_noisy = np.clip(image + gaussian, 0, 255).astype(np.uint8)

# Salt & Pepper Noise
def salt_pepper_noise(image, amount=0.08):
    noisy = image.copy()

    num_salt = int(amount * image.shape[0] * image.shape[1])

    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy

sp_noisy = salt_pepper_noise(image)

# Speckle Noise
noise = np.random.randn(*image.shape)
speckle_noisy = np.clip(image + image * noise * 0.4, 0, 255).astype(np.uint8)

# Poisson Noise
vals = 2 ** np.ceil(np.log2(len(np.unique(image))))
poisson_noisy = np.random.poisson(image * vals) / vals
poisson_noisy = np.clip(poisson_noisy, 0, 255).astype(np.uint8)

# Show all images in one window
titles = [
    "Original",
    "Gaussian Noise",
    "Salt & Pepper",
    "Speckle Noise",
    "Poisson Noise"
]

images = [
    image,
    gaussian_noisy,
    sp_noisy,
    speckle_noisy,
    poisson_noisy
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()