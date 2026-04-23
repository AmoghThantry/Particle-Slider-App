# Particle Slider App

An interactive Streamlit app for analyzing particle distributions across image sets. The app thresholds microscopy-style images, detects particle-like components, compares consistency across groups of five images, and summarizes how close each group is to an ideal distribution.

## What It Does

- Processes image groups named `1a.jpg` through `16e.jpg`
- Treats each numbered set as a 5-image group:
  `1a-1e`, `2a-2e`, ..., `16a-16e`
- Lets you adjust a live threshold slider to update analysis in real time
- Supports either:
  `pixel > threshold = particle`
  or
  `pixel < threshold = particle`
- Builds particle masks with basic morphology cleanup
- Detects connected components and ranks isolated, circular particles
- Computes per-image and per-group metrics, including:
  `particle area fraction`, `volume proxy`, `spacing match`, `uniformity`, and `perfect distribution %`
- Stitches the five originals and five masks together for easy visual comparison
- Exports a final comparison table as CSV

## Project Structure

```text
.
├── Particle_Slider_App.py
├── 1a.jpg ... 16e.jpg
└── README.md
```

The app looks for images in:

1. The same folder as `Particle_Slider_App.py`
2. Your `Downloads` folder

## How It Works

For each image, the app:

1. Loads the image with OpenCV
2. Converts it to grayscale
3. Applies a threshold to isolate particle regions
4. Cleans the mask with morphological open/close operations
5. Detects connected components
6. Filters components by size and circularity
7. Selects up to 25 isolated particles for size estimation
8. Computes summary metrics for particle coverage and distribution

For each 5-image group, it then:

1. Aligns the selected particle count across the five images
2. Calculates consistency statistics across the group
3. Builds a final comparison row for the overall dashboard

## Metrics Included

- `particle_pixels`
- `white_pixels`
- `particle_area_fraction`
- `detected_components`
- `selected_particle_count`
- `avg_particle_area_px`
- `avg_particle_equiv_diameter_px`
- `volume_proxy_px3`
- `spacing_match_pct`
- `uniformity_pct`
- `perfect_distribution_pct`

## Requirements

- Python 3.10+
- `streamlit`
- `opencv-python`
- `numpy`
- `pandas`

Install dependencies with:

```bash
pip install streamlit opencv-python numpy pandas
```

## Running The App

From the repo folder, run:

```bash
streamlit run Particle_Slider_App.py
```

Then open the local Streamlit URL shown in your terminal.

## Expected Image Naming

This project expects images to follow this pattern:

```text
1a.jpg  1b.jpg  1c.jpg  1d.jpg  1e.jpg
2a.jpg  2b.jpg  2c.jpg  2d.jpg  2e.jpg
...
16a.jpg 16b.jpg 16c.jpg 16d.jpg 16e.jpg
```

If your files use uppercase extensions like `.JPG`, the app will still try to find them.

## Why Streamlit

Streamlit makes this project useful as an exploration tool rather than just a batch script. You can tune the threshold interactively and immediately see how the masks, particle counts, and distribution metrics change across all 16 groups.

## Notes

- The app uses a 2D image-based size and volume proxy, not a true physical 3D measurement.
- Particle selection favors isolated, more circular components.
- Final comparison output can be downloaded directly as CSV from the UI.
