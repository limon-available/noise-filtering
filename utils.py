import base64
import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def load_image(uploaded_file) -> np.ndarray | None:
    """Load image from uploaded file or fallback to image.jpg."""
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    else:
        image = cv2.imread("image.jpg")
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


def display_image_with_title(image: np.ndarray, title: str, col):
    """Display an image with a title inside a Streamlit column."""
    col.image(image, use_container_width=True)
    col.markdown(
        f"<p style='text-align: center; font-weight: bold; font-size: 16px;'>{title}</p>",
        unsafe_allow_html=True,
    )


def display_image_with_metrics(image: np.ndarray, title: str, col, psnr: float = None, snr: float = None):
    """Display an image with a title and optional PSNR/SNR metrics below."""
    col.image(image, use_container_width=True)
    col.markdown(
        f"<p style='text-align: center; font-weight: bold; font-size: 15px;'>{title}</p>",
        unsafe_allow_html=True,
    )
    if psnr is not None and snr is not None:
        col.markdown(
            f"<p style='text-align: center; font-size: 13px; color: #2c3e50;'>"
            f"PSNR: {psnr:.2f} dB &nbsp;|&nbsp; SNR: {snr:.2f} dB</p>",
            unsafe_allow_html=True,
        )


def display_image_with_metrics_and_histogram(
    image: np.ndarray, title: str, col, psnr: float = None, snr: float = None
):
    """Display an image with title, PSNR/SNR metrics, and RGB histogram below."""
    col.image(image, use_container_width=True)
    col.markdown(
        f"<p style='text-align: center; font-weight: bold; font-size: 15px;'>{title}</p>",
        unsafe_allow_html=True,
    )
    if psnr is not None and snr is not None:
        col.markdown(
            f"<p style='text-align: center; font-size: 13px; color: #2c3e50;'>"
            f"PSNR: {psnr:.2f} dB &nbsp;|&nbsp; SNR: {snr:.2f} dB</p>",
            unsafe_allow_html=True,
        )
    hist_img = plot_histogram(image, f"{title} Histogram")
    col.image(hist_img, use_container_width=True)


def image_to_bytes(image: np.ndarray, fmt: str = "png") -> bytes:
    """Convert a NumPy image array to bytes for download."""
    success, buffer = cv2.imencode(f".{fmt}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        return b""
    return io.BytesIO(buffer.tobytes()).getvalue()


def get_image_download_link(image: np.ndarray, filename: str, label: str, fmt: str = "png") -> str:
    """Return an HTML anchor tag for downloading an image."""
    data = image_to_bytes(image, fmt)
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:image/{fmt};base64,{b64}" download="{filename}" style="text-decoration:none;">{label}</a>'
    return href


def plot_histogram(image: np.ndarray, title: str = "Histogram") -> np.ndarray:
    """Plot single grayscale histogram of an image and return as a NumPy array."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(hist, color="black", alpha=0.8, linewidth=1.5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim([0, 256])
    ax.set_xlabel("Pixel Intensity", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def display_image_with_histogram(image: np.ndarray, title: str, col):
    """Display an image and its histogram side by side in a column."""
    hist_img = plot_histogram(image, f"{title} Histogram")
    col.image(image, use_container_width=True)
    col.markdown(
        f"<p style='text-align: center; font-weight: bold; font-size: 15px;'>{title}</p>",
        unsafe_allow_html=True,
    )
    col.image(hist_img, use_container_width=True)
