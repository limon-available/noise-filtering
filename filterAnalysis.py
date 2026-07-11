import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================
# Read Image
# ==========================
image = cv2.imread("image.jpg")

if image is None:
    print("Image not found!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==========================
# Noise Functions
# ==========================

# Gaussian Noise
def add_gaussian_noise(image, mean=0, sigma=50):

    gaussian = np.random.normal(mean, sigma, image.shape)

    noisy = image + gaussian

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


# Salt & Pepper Noise
def add_salt_pepper_noise(image, amount=0.08):

    noisy = image.copy()

    total_pixels = int(amount * image.shape[0] * image.shape[1])

    # Salt
    coords = [np.random.randint(0, i, total_pixels // 2)
              for i in image.shape[:2]]

    noisy[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i, total_pixels // 2)
              for i in image.shape[:2]]

    noisy[coords[0], coords[1]] = 0

    return noisy


# Speckle Noise
def add_speckle_noise(image):

    noise = np.random.randn(*image.shape)

    noisy = image + image * noise * 0.4

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


# Poisson Noise
def add_poisson_noise(image):

    vals = len(np.unique(image))

    vals = 2 ** np.ceil(np.log2(vals))

    noisy = np.random.poisson(image * vals) / float(vals)

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


# ==========================
# SNR Function
# ==========================

def calculate_snr(original, processed):

    original = original.astype(np.float64)

    processed = processed.astype(np.float64)

    signal_power = np.mean(original ** 2)

    noise_power = np.mean((original - processed) ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)

# ==========================
# Filter Analysis Function
# ==========================

def analyze_filter(original, noisy, filter_name, iterations=5):

    images = [noisy]
    psnr_values = []
    snr_values = []

    current = noisy.copy()

    for i in range(iterations):

        if filter_name == "gaussian":
            current = cv2.GaussianBlur(current, (5, 5), 0)

        elif filter_name == "median":
            current = cv2.medianBlur(current, 5)

        elif filter_name == "bilateral":
            current = cv2.bilateralFilter(current, 9, 75, 75)

        images.append(current.copy())

        psnr = cv2.PSNR(original, current)
        snr = calculate_snr(original, current)

        psnr_values.append(psnr)
        snr_values.append(snr)

    return images, psnr_values, snr_values


# ==========================
# Generate Noisy Images
# ==========================

gaussian_noisy = add_gaussian_noise(image)

sp_noisy = add_salt_pepper_noise(image)

speckle_noisy = add_speckle_noise(image)

poisson_noisy = add_poisson_noise(image)


# ==========================
# Run Analysis
# ==========================

gaussian_images, gaussian_psnr, gaussian_snr = analyze_filter(
    image,
    gaussian_noisy,
    "gaussian"
)

sp_images, sp_psnr, sp_snr = analyze_filter(
    image,
    sp_noisy,
    "median"
)

speckle_images, speckle_psnr, speckle_snr = analyze_filter(
    image,
    speckle_noisy,
    "bilateral"
)

poisson_images, poisson_psnr, poisson_snr = analyze_filter(
    image,
    poisson_noisy,
    "gaussian"
)

# =====================================
# Show Everything in One Window
# =====================================

all_results = [
    ("Gaussian", gaussian_images),
    ("Salt & Pepper", sp_images),
    ("Speckle", speckle_images),
    ("Poisson", poisson_images)
]

fig, axes = plt.subplots(4, 7, figsize=(26, 16))

column_titles = [
    "Original",
    "Noisy",
    "Filter-1",
    "Filter-2",
    "Filter-3",
    "Filter-4",
    "Filter-5"
]

# Column Titles
for j in range(7):
    axes[0, j].set_title(column_titles[j], fontsize=12)

# Fill Images
# Fill Images
for i, (noise_name, imgs) in enumerate(all_results):

    display_images = [image] + imgs

    for j in range(7):

        axes[i, j].imshow(display_images[j])
        axes[i, j].axis("off")

        # Original এবং Noisy এর নিচে কিছু লিখবে না
        if j >= 2:

            if noise_name == "Gaussian":
                psnr = gaussian_psnr[j-2]
                snr = gaussian_snr[j-2]

            elif noise_name == "Salt & Pepper":
                psnr = sp_psnr[j-2]
                snr = sp_snr[j-2]

            elif noise_name == "Speckle":
                psnr = speckle_psnr[j-2]
                snr = speckle_snr[j-2]

            elif noise_name == "Poisson":
                psnr = poisson_psnr[j-2]
                snr = poisson_snr[j-2]

            axes[i, j].text(
                0.5,
                -0.18,
                f"PSNR: {psnr:.2f}\nSNR : {snr:.2f}",
                fontsize=8,
                ha="center",
                va="top",
                transform=axes[i, j].transAxes
            )

    # Row Label
    axes[i, 0].text(
        -0.45,
        0.5,
        noise_name,
        fontsize=12,
        fontweight="bold",
        rotation=90,
        ha="center",
        va="center",
        transform=axes[i, 0].transAxes
    )

plt.suptitle(
    "Image Filtering Analysis",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# =====================================
# Print PSNR & SNR
# =====================================

results = [
    ("Gaussian", gaussian_psnr, gaussian_snr),
    ("Salt & Pepper", sp_psnr, sp_snr),
    ("Speckle", speckle_psnr, speckle_snr),
    ("Poisson", poisson_psnr, poisson_snr)
]

for name, psnr, snr in results:

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Iteration\tPSNR(dB)\tSNR(dB)")

    for i in range(5):

        print(f"{i+1}\t\t{psnr[i]:.2f}\t\t{snr[i]:.2f}")