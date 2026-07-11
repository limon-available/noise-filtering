import pandas as pd
import streamlit as st

from noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
    add_poisson_noise,
)
from filtering import analyze_filter
from utils import (
    load_image,
    display_image_with_title,
    display_image_with_metrics,
    display_image_with_histogram,
    display_image_with_metrics_and_histogram,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Noise Filtering",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .stApp { background-color: #f5f7fa; }
    .block-container { padding-top: 1.5rem; }

    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: -0.5px;
        margin-bottom: 0.15rem;
    }
    .main-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.2rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.6rem;
    }
    .sidebar-header {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.4rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
    }
    .st-emotion-cache-1y4p8pa { padding-top: 1rem; }
    hr { margin: 0.8rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-size:1.4rem; font-weight:700; color:#1a1a2e; margin-bottom:0.2rem;'>⚙️ Controls</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Shared: Upload
    with st.container(border=True):
        st.markdown('<div class="sidebar-header">📁 Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            label_visibility="collapsed",
            key="global_uploader",
        )
        if uploaded_file is None:
            st.info("ℹ️ Using default **image.jpg**")
        else:
            st.success(f"✅ **{uploaded_file.name}**")

    st.markdown("---")

    # Shared: Histogram toggle
    with st.container(border=True):
        st.markdown('<div class="sidebar-header">📊 Show Histograms</div>', unsafe_allow_html=True)
        show_histograms = st.checkbox(
            "Display RGB histograms below images",
            value=False,
            label_visibility="collapsed",
            key="show_histograms",
        )

    st.markdown("---")

    # Navigation
    mode = st.radio(
        "Mode",
        options=["🎨 Noise Generation", "🔬 Filtering Analysis"],
        index=0,
        label_visibility="collapsed",
        key="mode",
    )

    st.markdown("---")

    # ── Tab-specific controls ──
    if mode == "🎨 Noise Generation":
        with st.container(border=True):
            st.markdown('<div class="sidebar-header">🎯 Noise Type</div>', unsafe_allow_html=True)
            noise_option = st.radio(
                "Select noise type",
                options=["Show All", "Gaussian", "Salt & Pepper", "Speckle", "Poisson"],
                index=0,
                label_visibility="collapsed",
                key="noise_option",
            )

    else:  # Filtering Analysis
        with st.container(border=True):
            st.markdown('<div class="sidebar-header">🎯 Noise Type</div>', unsafe_allow_html=True)
            filter_noise_type = st.radio(
                "Select noise type",
                options=["Gaussian", "Salt & Pepper", "Speckle", "Poisson"],
                index=0,
                label_visibility="collapsed",
                key="filter_noise_type",
            )

        with st.container(border=True):
            st.markdown('<div class="sidebar-header">🔧 Filter</div>', unsafe_allow_html=True)
            filter_type = st.radio(
                "Select filter",
                options=[
                    "gaussian",
                    "median",
                    "bilateral",
                    "box",
                    "weighted_smoothing",
                    "sobel",
                    "laplacian",
                    "unsharp_masking",
                    "high_boost",
                ],
                index=0,
                label_visibility="collapsed",
                key="filter_type",
            )

        with st.container(border=True):
            st.markdown('<div class="sidebar-header">🔁 Iterations</div>', unsafe_allow_html=True)
            iterations = st.slider(
                "Number of iterations",
                min_value=1,
                max_value=10,
                value=5,
                label_visibility="collapsed",
                key="iterations",
            )

        run_button = st.button("▶️ Run Analysis", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────
# Load Image
# ─────────────────────────────────────────────────────────
image = load_image(uploaded_file)

if image is None:
    st.error("❌ Could not load image. Make sure `image.jpeg` exists or upload a valid file.")
    st.stop()

# ─────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎛️ Image Noise Filtering</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Upload an image, add noise, then apply iterative filters and track quality metrics</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# =====================================================================
# NOISE GENERATION
# =====================================================================
if mode == "🎨 Noise Generation":
    with st.spinner("Generating noise..."):
        gaussian_img = add_gaussian_noise(image)
        sp_img = add_salt_pepper_noise(image)
        speckle_img = add_speckle_noise(image)
        poisson_img = add_poisson_noise(image)

    st.markdown('<div class="section-header">📸 Results</div>', unsafe_allow_html=True)

    display_fn = display_image_with_histogram if show_histograms else display_image_with_title

    if noise_option == "Show All":
        with st.container(border=True):
            st.markdown("**Row 1 — Original & Two Noise Types**")
            col1, col2, col3 = st.columns(3)
            display_fn(image, "🟢 Original", col1)
            display_fn(gaussian_img, "🌫️ Gaussian", col2)
            display_fn(sp_img, "🧂 Salt & Pepper", col3)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("**Row 2 — Two Noise Types**")
            col4, col5 = st.columns(2)
            display_fn(speckle_img, "✨ Speckle", col4)
            display_fn(poisson_img, "📊 Poisson", col5)

    else:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            display_fn(image, "🟢 Original", col1)

            noise_map = {
                "Gaussian": ("🌫️ Gaussian", gaussian_img),
                "Salt & Pepper": ("🧂 Salt & Pepper", sp_img),
                "Speckle": ("✨ Speckle", speckle_img),
                "Poisson": ("📊 Poisson", poisson_img),
            }
            title, noisy_img = noise_map[noise_option]
            display_fn(noisy_img, title, col2)

# =====================================================================
# FILTERING ANALYSIS
# =====================================================================
else:
    with st.spinner("Generating noisy image..."):
        noise_map = {
            "Gaussian": add_gaussian_noise,
            "Salt & Pepper": add_salt_pepper_noise,
            "Speckle": add_speckle_noise,
            "Poisson": add_poisson_noise,
        }
        noisy_img = noise_map[filter_noise_type](image)

    if run_button:
        with st.spinner(f"Applying **{filter_type.capitalize()}** filter ({iterations} iterations)..."):
            filtered_images, psnr_values, snr_values = analyze_filter(
                original=image,
                noisy=noisy_img,
                filter_name=filter_type,
                iterations=iterations,
            )

        st.markdown('<div class="section-header">📸 Results</div>', unsafe_allow_html=True)

        display_fn = display_image_with_histogram if show_histograms else display_image_with_title

        with st.container(border=True):
            st.markdown("**Original & Noisy**")
            col1, col2 = st.columns(2)
            display_fn(image, "🟢 Original", col1)
            display_fn(noisy_img, f"📛 Noisy ({filter_noise_type})", col2)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f"**Filtered Images — {filter_type.capitalize()} Filter**")
            num_filtered = len(filtered_images) - 1

            metrics_fn = display_image_with_metrics_and_histogram if show_histograms else display_image_with_metrics

            for row_start in range(0, num_filtered, 5):
                row_end = min(row_start + 5, num_filtered)
                cols = st.columns(5)

                for j in range(row_start, row_end):
                    img_idx = j + 1
                    metrics_fn(
                        filtered_images[img_idx],
                        f"Filter-{img_idx}",
                        cols[j - row_start],
                        psnr=psnr_values[j],
                        snr=snr_values[j],
                    )

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 View Metrics Table", expanded=True):
            df = pd.DataFrame(
                {
                    "Iteration": list(range(1, iterations + 1)),
                    "PSNR (dB)": [f"{v:.2f}" for v in psnr_values],
                    "SNR (dB)": [f"{v:.2f}" for v in snr_values],
                }
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        st.info("👈 Configure the settings in the sidebar and click **Run Analysis** to see results.")