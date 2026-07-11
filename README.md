# 🎛️ Image Noise Filtering

A Streamlit-based application for adding various types of noise to images and applying iterative filtering with quality metrics tracking.

## Features

### 🎨 Noise Generation
Add four types of noise to any image:
- **Gaussian Noise** — Additive Gaussian noise (mean=0, sigma=50)
- **Salt & Pepper Noise** — Random white and black pixels (amount=0.08)
- **Speckle Noise** — Multiplicative speckle noise (factor=0.4)
- **Poisson Noise** — Poisson-distributed noise

### 🔬 Filtering Analysis
Apply iterative filtering and track quality metrics:
- **Gaussian Filter** — `cv2.GaussianBlur` with 5×5 kernel
- **Median Filter** — `cv2.medianBlur` with 5×5 kernel
- **Bilateral Filter** — `cv2.bilateralFilter` (d=9, sigmaColor=75, sigmaSpace=75)

Each filter can be applied repeatedly (1–10 iterations). PSNR and SNR are calculated after every iteration and displayed in a metrics table.

## Project Structure

```
noise-filtering/
├── app.py              # Streamlit UI (2 tabs)
├── noise.py            # Noise generation functions
├── filters.py          # Filter functions (Gaussian, Median, Bilateral)
├── metrics.py          # PSNR and SNR calculation
├── filtering.py        # Iterative filtering analysis
├── utils.py            # Image loading and display helpers
├── noiseAdd.py         # Original noise script (untouched)
├── filterAnalysis.py   # Original filter analysis script (untouched)
├── image.jpeg          # Default fallback image
├── requirements.txt    # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/limon-available/noise-filtering.git
   cd noise-filtering
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at **http://localhost:8501**

## Usage

1. **Upload an image** via the sidebar (or use the default `image.jpeg`)
2. **Noise Generation tab**: Select a noise type or "Show All" to preview all four
3. **Filtering Analysis tab**: Choose noise type, filter, and iteration count, then click **Run Analysis**
4. View PSNR and SNR metrics in the expandable table

## Dependencies

- streamlit
- opencv-python-headless
- numpy
- pandas