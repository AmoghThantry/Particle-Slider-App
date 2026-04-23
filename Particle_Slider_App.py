from pathlib import Path
import math

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# FILE SETUP
# ============================================================
APP_DIR = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
IMAGE_SEARCH_DIRS = [APP_DIR, DOWNLOADS]

GROUP_COUNT = 16
GROUP_SUFFIXES = ["a", "b", "c", "d", "e"]
IMAGE_GROUPS = {
    f"Group {group_num}": [f"{group_num}{suffix}.jpg" for suffix in GROUP_SUFFIXES]
    for group_num in range(1, GROUP_COUNT + 1)
}

TARGET_PARTICLE_COUNT = 25
MIN_COMPONENT_AREA = 25
MAX_COMPONENT_AREA_FRACTION = 0.20
MIN_CIRCULARITY = 0.45
MORPH_KERNEL = 3
GRID_ROWS = 6
GRID_COLS = 6


# ============================================================
# HELPERS
# ============================================================
def resolve_image_path(name: str) -> Path:
    candidates = []

    for base_dir in IMAGE_SEARCH_DIRS:
        p = base_dir / name

        if p.suffix:
            candidates.append(p)
        else:
            candidates.extend([
                base_dir / f"{name}.jpg",
                base_dir / f"{name}.JPG",
                base_dir / f"{name}.jpeg",
                base_dir / f"{name}.JPEG",
                base_dir / f"{name}.png",
                base_dir / f"{name}.PNG",
            ])

    for c in candidates:
        if c.exists():
            return c

    search_dirs = ", ".join(str(path) for path in IMAGE_SEARCH_DIRS)
    raise FileNotFoundError(f"Could not find image for {name} in: {search_dirs}")


@st.cache_data(show_spinner=False)
def load_image(path_str: str):
    img = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {path_str}")
    return img


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_to_rgb(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def make_particle_mask(img_bgr, threshold_value=245, brighter_is_particle=True):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if brighter_is_particle:
        raw_mask = (gray > threshold_value).astype(np.uint8) * 255
    else:
        raw_mask = (gray < threshold_value).astype(np.uint8) * 255

    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


def connected_components(mask):
    return cv2.connectedComponentsWithStats(mask, connectivity=8)


def component_circularity(area, perimeter):
    if perimeter <= 0:
        return 0.0
    return (4.0 * math.pi * area) / (perimeter ** 2)


def extract_components(mask, min_area=25, max_area_fraction=0.20):
    h, w = mask.shape
    image_area = h * w
    max_area = image_area * max_area_fraction

    num_labels, labels, stats, centroids = connected_components(mask)

    rows = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        comp_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(cnt, True))
        circularity = component_circularity(area, perimeter)
        eq_diameter = math.sqrt(4 * area / math.pi)

        rows.append({
            "label_id": i,
            "area_px": area,
            "centroid_x_px": float(centroids[i][0]),
            "centroid_y_px": float(centroids[i][1]),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": bw,
            "bbox_h": bh,
            "perimeter_px": perimeter,
            "circularity": circularity,
            "equiv_diameter_px": eq_diameter,
        })

    return pd.DataFrame(rows)


def select_isolated_particles(components_df, target_count=25, min_circularity=0.45):
    if components_df.empty:
        return components_df.copy()

    df = components_df.copy()
    pts = df[["centroid_x_px", "centroid_y_px"]].to_numpy()

    if len(df) >= 2:
        dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        dmat[dmat == 0] = np.inf
        nn = dmat.min(axis=1)
    else:
        nn = np.array([np.nan])

    df["nearest_neighbor_px"] = nn
    df["isolation_score"] = df["nearest_neighbor_px"].fillna(0)
    df["usable"] = df["circularity"] >= min_circularity

    usable = df[df["usable"]].copy()
    if usable.empty:
        usable = df.copy()

    usable = usable.sort_values(
        by=["circularity", "isolation_score", "area_px"],
        ascending=[False, False, True]
    )

    return usable.head(target_count).copy()


def overlay_centroids(img_bgr, particles_df):
    out = img_bgr.copy()
    for _, row in particles_df.iterrows():
        x = int(round(row["centroid_x_px"]))
        y = int(round(row["centroid_y_px"]))
        cv2.circle(out, (x, y), 4, (0, 0, 255), -1)
    return out


def grid_distribution_stats(mask, rows=6, cols=6):
    h, w = mask.shape
    cell_h = h / rows
    cell_w = w / cols

    cover = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            y0 = int(round(r * cell_h))
            y1 = int(round((r + 1) * cell_h))
            x0 = int(round(c * cell_w))
            x1 = int(round((c + 1) * cell_w))
            patch = mask[y0:y1, x0:x1]
            cover[r, c] = np.count_nonzero(patch)

    flat = cover.flatten()
    mean_cover = float(np.mean(flat))
    std_cover = float(np.std(flat))
    cv_cover = float(std_cover / mean_cover) if mean_cover > 0 else np.nan

    return cover, mean_cover, std_cover, cv_cover


def compute_image_metrics(image_path: Path, threshold_value: int, brighter_is_particle: bool):
    img = load_image(str(image_path))
    h, w = img.shape[:2]
    total_px = h * w

    particle_mask = make_particle_mask(
        img,
        threshold_value=threshold_value,
        brighter_is_particle=brighter_is_particle
    )

    white_mask = cv2.bitwise_not(particle_mask)

    components = extract_components(
        particle_mask,
        min_area=MIN_COMPONENT_AREA,
        max_area_fraction=MAX_COMPONENT_AREA_FRACTION
    )

    selected = select_isolated_particles(
        components,
        target_count=TARGET_PARTICLE_COUNT,
        min_circularity=MIN_CIRCULARITY
    )

    particle_pixels = int(np.count_nonzero(particle_mask))
    white_pixels = total_px - particle_pixels

    particle_fraction = particle_pixels / total_px
    white_fraction = white_pixels / total_px

    if selected.empty:
        avg_area_px = np.nan
        avg_equiv_diameter_px = np.nan
        particle_count_used = 0
    else:
        avg_area_px = float(selected["area_px"].mean())
        avg_equiv_diameter_px = float(selected["equiv_diameter_px"].mean())
        particle_count_used = int(len(selected))

    volume_proxy_px3 = np.nan if np.isnan(avg_equiv_diameter_px) else particle_pixels * avg_equiv_diameter_px

    ideal_particle_count = np.nan if (np.isnan(avg_area_px) or avg_area_px <= 0) else particle_pixels / avg_area_px

    actual_particle_count = max(len(components), 1)
    actual_spacing_proxy = white_pixels / actual_particle_count
    ideal_spacing_proxy = np.nan if (np.isnan(ideal_particle_count) or ideal_particle_count <= 0) else white_pixels / ideal_particle_count

    if np.isnan(ideal_spacing_proxy) or ideal_spacing_proxy == 0:
        spacing_match_pct = np.nan
    else:
        spacing_match_pct = 100.0 * min(actual_spacing_proxy, ideal_spacing_proxy) / max(actual_spacing_proxy, ideal_spacing_proxy)

    _, grid_mean, grid_std, grid_cv = grid_distribution_stats(
        particle_mask,
        rows=GRID_ROWS,
        cols=GRID_COLS
    )

    uniformity_pct = np.nan if np.isnan(grid_cv) else max(0.0, 100.0 * (1.0 - min(grid_cv, 1.5) / 1.5))

    if np.isnan(spacing_match_pct) and np.isnan(uniformity_pct):
        perfect_distribution_pct = np.nan
    elif np.isnan(spacing_match_pct):
        perfect_distribution_pct = uniformity_pct
    elif np.isnan(uniformity_pct):
        perfect_distribution_pct = spacing_match_pct
    else:
        perfect_distribution_pct = 0.5 * spacing_match_pct + 0.5 * uniformity_pct

    overlay = overlay_centroids(img, selected)

    summary = {
        "image_name": image_path.name,
        "particle_pixels": particle_pixels,
        "white_pixels": white_pixels,
        "particle_area_fraction": round(particle_fraction, 6),
        "white_area_fraction": round(white_fraction, 6),
        "detected_components": int(len(components)),
        "selected_particle_count": particle_count_used,
        "avg_particle_area_px": round(avg_area_px, 3) if not np.isnan(avg_area_px) else np.nan,
        "avg_particle_equiv_diameter_px": round(avg_equiv_diameter_px, 3) if not np.isnan(avg_equiv_diameter_px) else np.nan,
        "volume_proxy_px3": round(volume_proxy_px3, 3) if not np.isnan(volume_proxy_px3) else np.nan,
        "ideal_particle_count": round(ideal_particle_count, 3) if not np.isnan(ideal_particle_count) else np.nan,
        "actual_spacing_proxy_px2_per_particle": round(actual_spacing_proxy, 3),
        "ideal_spacing_proxy_px2_per_particle": round(ideal_spacing_proxy, 3) if not np.isnan(ideal_spacing_proxy) else np.nan,
        "spacing_match_pct": round(spacing_match_pct, 3) if not np.isnan(spacing_match_pct) else np.nan,
        "grid_mean_particle_pixels": round(grid_mean, 3) if not np.isnan(grid_mean) else np.nan,
        "grid_std_particle_pixels": round(grid_std, 3) if not np.isnan(grid_std) else np.nan,
        "grid_cv": round(grid_cv, 4) if not np.isnan(grid_cv) else np.nan,
        "uniformity_pct": round(uniformity_pct, 3) if not np.isnan(uniformity_pct) else np.nan,
        "perfect_distribution_pct": round(perfect_distribution_pct, 3) if not np.isnan(perfect_distribution_pct) else np.nan,
    }

    return {
        "img_bgr": img,
        "particle_mask": particle_mask,
        "white_mask": white_mask,
        "overlay_bgr": overlay,
        "components_df": components,
        "selected_df": selected,
        "summary": summary,
    }


def analyze_group(group_name, file_names, threshold_value, brighter_is_particle):
    paths = [resolve_image_path(name) for name in file_names]
    results = [compute_image_metrics(p, threshold_value, brighter_is_particle) for p in paths]

    selected_counts = [r["summary"]["selected_particle_count"] for r in results]
    common_n = min(TARGET_PARTICLE_COUNT, min(selected_counts)) if selected_counts else 0

    for r in results:
        r["selected_df"] = r["selected_df"].head(common_n).copy()
        if common_n > 0:
            r["summary"]["selected_particle_count"] = common_n
            r["summary"]["avg_particle_area_px"] = round(float(r["selected_df"]["area_px"].mean()), 3)
            r["summary"]["avg_particle_equiv_diameter_px"] = round(float(r["selected_df"]["equiv_diameter_px"].mean()), 3)
            r["summary"]["volume_proxy_px3"] = round(
                r["summary"]["particle_pixels"] * r["summary"]["avg_particle_equiv_diameter_px"], 3
            )
        else:
            r["summary"]["selected_particle_count"] = 0

    df = pd.DataFrame([r["summary"] for r in results])

    consistency = {
        "group": group_name,
        "common_selected_particle_count": common_n,
        "mean_perfect_distribution_pct": round(df["perfect_distribution_pct"].mean(), 3),
        "std_perfect_distribution_pct": round(df["perfect_distribution_pct"].std(ddof=0), 3),
        "cv_perfect_distribution_pct": round(
            df["perfect_distribution_pct"].std(ddof=0) / df["perfect_distribution_pct"].mean(), 4
        ) if df["perfect_distribution_pct"].mean() > 0 else np.nan,
        "mean_volume_proxy_px3": round(df["volume_proxy_px3"].mean(), 3),
        "std_volume_proxy_px3": round(df["volume_proxy_px3"].std(ddof=0), 3),
        "cv_volume_proxy_px3": round(
            df["volume_proxy_px3"].std(ddof=0) / df["volume_proxy_px3"].mean(), 4
        ) if df["volume_proxy_px3"].mean() > 0 else np.nan,
        "mean_particle_area_fraction": round(df["particle_area_fraction"].mean(), 6),
        "std_particle_area_fraction": round(df["particle_area_fraction"].std(ddof=0), 6),
        "cv_particle_area_fraction": round(
            df["particle_area_fraction"].std(ddof=0) / df["particle_area_fraction"].mean(), 4
        ) if df["particle_area_fraction"].mean() > 0 else np.nan,
        "mean_avg_particle_equiv_diameter_px": round(df["avg_particle_equiv_diameter_px"].mean(), 3),
        "mean_spacing_match_pct": round(df["spacing_match_pct"].mean(), 3),
        "mean_uniformity_pct": round(df["uniformity_pct"].mean(), 3),
    }

    return results, df, pd.DataFrame([consistency]), common_n


def stitch_with_partitions(images_rgb, line_color=(255, 0, 0), line_width=6):
    heights = [img.shape[0] for img in images_rgb]
    max_h = max(heights)

    resized = []
    for img in images_rgb:
        h, w = img.shape[:2]
        scale = max_h / h
        new_w = int(round(w * scale))
        resized.append(cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA))

    total_w = sum(img.shape[1] for img in resized) + line_width * (len(resized) - 1)
    canvas = np.ones((max_h, total_w, 3), dtype=np.uint8) * 255

    x = 0
    for i, img in enumerate(resized):
        w = img.shape[1]
        canvas[:, x:x+w] = img
        x += w
        if i < len(resized) - 1:
            canvas[:, x:x+line_width] = np.array(line_color, dtype=np.uint8)
            x += line_width

    return canvas


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(layout="wide", page_title="Particle Slider App")

st.title("Particle Slider App")
st.write("Adjust the threshold slider and all masks, metrics, and comparisons update together.")

threshold_value = st.slider("Threshold value", min_value=0, max_value=255, value=245, step=1)
brighter_is_particle = st.radio(
    "Particle rule",
    options=[True, False],
    format_func=lambda x: "Greater than threshold = particle" if x else "Less than threshold = particle",
    index=0
)

st.caption("True particle height is not available from plain 2D images, so this app uses a consistent size / volume proxy.")

try:
    analyzed_groups = []
    for group_name, file_names in IMAGE_GROUPS.items():
        results, metrics_df, consistency_df, common_n = analyze_group(
            group_name,
            file_names,
            threshold_value,
            brighter_is_particle
        )
        analyzed_groups.append({
            "group_name": group_name,
            "results": results,
            "metrics_df": metrics_df,
            "consistency_df": consistency_df,
            "common_n": common_n,
        })
except Exception as e:
    st.error(str(e))
    st.stop()

common_counts_text = ", ".join(
    f'{group["group_name"]} = {group["common_n"]}'
    for group in analyzed_groups
)
st.info(f"Common isolated particle counts used per group: {common_counts_text}")

for group in analyzed_groups:
    group_name = group["group_name"]
    results = group["results"]
    st.subheader(group_name)

    originals = stitch_with_partitions([bgr_to_rgb(r["img_bgr"]) for r in results])
    masks = stitch_with_partitions([mask_to_rgb(r["particle_mask"]) for r in results])

    c1, c2 = st.columns(2)
    with c1:
        st.image(originals, caption=f"{group_name} originals stitched", use_container_width=True)
    with c2:
        st.image(masks, caption=f"{group_name} toleranced masks stitched", use_container_width=True)

st.subheader("Per-image metrics")
tabs = st.tabs([group["group_name"] for group in analyzed_groups])

for tab, group in zip(tabs, analyzed_groups):
    with tab:
        st.dataframe(group["metrics_df"], use_container_width=True)
        st.dataframe(group["consistency_df"], use_container_width=True)

final_compare = pd.DataFrame([
    {
        "group": group["group_name"],
        "mean_%_perfect_distribution": group["consistency_df"].iloc[0]["mean_perfect_distribution_pct"],
        "perfect_target_%": 100.0,
        "gap_to_perfect_%": 100.0 - group["consistency_df"].iloc[0]["mean_perfect_distribution_pct"],
        "mean_volume_proxy_px3": group["consistency_df"].iloc[0]["mean_volume_proxy_px3"],
        "volume_cv_across_5": group["consistency_df"].iloc[0]["cv_volume_proxy_px3"],
        "distribution_cv_across_5": group["consistency_df"].iloc[0]["cv_perfect_distribution_pct"],
    }
    for group in analyzed_groups
])

st.subheader("Final comparison to perfect distribution")
st.dataframe(final_compare, use_container_width=True)

csv = final_compare.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download final comparison CSV",
    data=csv,
    file_name="Particle_Slider_App_final_comparison.csv",
    mime="text/csv"
)
