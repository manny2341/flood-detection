"""
flood_engine.py — Core flood detection pipeline
Sentinel-1 SAR change detection using log-ratio + Otsu thresholding
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from scipy.ndimage import uniform_filter, variance as ndimage_variance
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Preset flood events (real coordinates)
# ─────────────────────────────────────────────
FLOOD_EVENTS = {
    "Pakistan 2022": {
        "description": "Catastrophic monsoon floods — one-third of Pakistan submerged. ~33M people affected.",
        "bbox": [66.5, 26.5, 69.5, 29.5],
        "date_pre": "2022-07-01",
        "date_post": "2022-09-01",
        "severity": "Severe",
        "country": "Pakistan",
    },
    "Libya 2023": {
        "description": "Storm Daniel triggered catastrophic flash floods in Derna — city largely destroyed.",
        "bbox": [21.5, 32.0, 23.5, 33.5],
        "date_pre": "2023-08-01",
        "date_post": "2023-09-15",
        "severity": "Severe",
        "country": "Libya",
    },
    "Nigeria 2022": {
        "description": "Worst flooding in a decade — displaced 1.4M people across 33 of 36 states.",
        "bbox": [6.0, 6.0, 8.5, 8.5],
        "date_pre": "2022-08-01",
        "date_post": "2022-10-15",
        "severity": "Moderate",
        "country": "Nigeria",
    },
    "Bangladesh 2022": {
        "description": "Severe flash floods in Sylhet division — worst flooding in 20 years.",
        "bbox": [91.0, 24.0, 92.5, 25.5],
        "date_pre": "2022-05-01",
        "date_post": "2022-06-20",
        "severity": "Moderate",
        "country": "Bangladesh",
    },
    "Australia 2022": {
        "description": "Queensland and NSW flooding — worst event in 50 years, Brisbane inundated.",
        "bbox": [152.5, -28.0, 154.0, -26.5],
        "date_pre": "2022-01-01",
        "date_post": "2022-03-01",
        "severity": "Severe",
        "country": "Australia",
    },
}


# ─────────────────────────────────────────────
# Synthetic SAR scene generator
# ─────────────────────────────────────────────

def _lee_filter(img, size=7):
    """Lee speckle filter for SAR imagery."""
    img_mean = uniform_filter(img, size)
    img_sq_mean = uniform_filter(img ** 2, size)
    img_variance = img_sq_mean - img_mean ** 2
    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance + 1e-10)
    return img_mean + img_weights * (img - img_mean)


def generate_sar_scene(event_name, scene_type="pre", seed=42):
    """
    Generate a realistic synthetic Sentinel-1 SAR scene (VV band, linear scale).
    scene_type: 'pre' or 'post' (post has flood regions added)
    Returns: (array float32, GeoTIFF transform, CRS)
    """
    rng = np.random.default_rng(seed)
    rows, cols = 512, 512
    event = FLOOD_EVENTS[event_name]
    bbox = event["bbox"]

    # Base land backscatter (~-12 dB → linear ~0.063)
    scene = rng.gamma(shape=2.0, scale=0.03, size=(rows, cols)).astype(np.float32)

    # Add terrain structure — rolling hills / river valleys
    x = np.linspace(0, 4 * np.pi, cols)
    y = np.linspace(0, 4 * np.pi, rows)
    xx, yy = np.meshgrid(x, y)
    terrain = (0.015 * np.sin(xx * 0.7) * np.cos(yy * 0.5) +
               0.010 * np.sin(xx * 1.3 + yy * 0.9)).astype(np.float32)
    scene += terrain - terrain.min()

    # Permanent water bodies (rivers/lakes — low backscatter ~-20 dB → ~0.010)
    # River running diagonally
    river_mask = np.zeros((rows, cols), bool)
    for r in range(rows):
        col_centre = int(cols * 0.35 + r * 0.15) % cols
        river_mask[r, max(0, col_centre - 6):col_centre + 6] = True
    # Lake in lower-left quadrant
    cy, cx = int(rows * 0.7), int(cols * 0.25)
    yr, xr = np.ogrid[:rows, :cols]
    lake_mask = ((yr - cy) ** 2 / 30 ** 2 + (xr - cx) ** 2 / 45 ** 2) < 1
    water_mask = river_mask | lake_mask
    scene[water_mask] = rng.gamma(shape=1.5, scale=0.006, size=water_mask.sum()).astype(np.float32)

    # Urban area (high backscatter ~-5 dB → ~0.316) top-right
    urban_r = slice(int(rows * 0.05), int(rows * 0.25))
    urban_c = slice(int(cols * 0.70), int(cols * 0.95))
    scene[urban_r, urban_c] += rng.gamma(shape=3.0, scale=0.08, size=(
        urban_r.stop - urban_r.start, urban_c.stop - urban_c.start
    )).astype(np.float32)

    if scene_type == "post":
        # Flood zones — large contiguous areas newly covered in water
        # Main flood plain (centre-left)
        flood1 = np.zeros((rows, cols), bool)
        flood1[int(rows*0.35):int(rows*0.65), int(cols*0.20):int(cols*0.55)] = True
        # Overflow along river
        flood2 = np.zeros((rows, cols), bool)
        for r in range(int(rows*0.4), int(rows*0.85)):
            col_centre = int(cols * 0.35 + r * 0.15) % cols
            flood2[r, max(0, col_centre - 20):col_centre + 20] = True
        # Smaller flood pocket
        flood3 = np.zeros((rows, cols), bool)
        flood3[int(rows*0.60):int(rows*0.78), int(cols*0.55):int(cols*0.72)] = True

        flood_mask = (flood1 | flood2 | flood3) & ~water_mask
        scene[flood_mask] = rng.gamma(
            shape=1.5, scale=0.007, size=flood_mask.sum()
        ).astype(np.float32)

    # SAR speckle noise (multiplicative)
    speckle = rng.exponential(scale=1.0, size=(rows, cols)).astype(np.float32)
    scene = scene * speckle

    # Clip to realistic range
    scene = np.clip(scene, 0, 1).astype(np.float32)

    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], cols, rows)
    crs = CRS.from_epsg(4326)
    return scene, transform, crs


# ─────────────────────────────────────────────
# Save / load GeoTIFF helpers
# ─────────────────────────────────────────────

def save_geotiff(array, path, transform, crs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1],
        count=1, dtype=array.dtype,
        crs=crs, transform=transform,
    ) as dst:
        dst.write(array, 1)


def load_geotiff(path):
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    # Mask nodata sentinel values (e.g. -32768 from Planetary Computer)
    nodata_mask = (array < -1000) | np.isnan(array) | np.isinf(array)
    if nodata_mask.any():
        # Replace nodata with median of valid pixels
        valid = array[~nodata_mask]
        fill = float(np.median(valid)) if len(valid) > 0 else 0.0
        array[nodata_mask] = fill

    # Clip to valid SAR linear backscatter range (0 to ~10)
    array = np.clip(array, 0.0, 10.0)
    return array, transform, crs


# ─────────────────────────────────────────────
# Core detection pipeline
# ─────────────────────────────────────────────

def load_real_data(event_name, data_dir=None):
    """
    Load real downloaded Sentinel-1 GeoTIFFs for an event.
    Returns (pre_arr, post_arr, transform, crs) or None if not downloaded.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
    safe_name = event_name.replace(" ", "_")
    pre_path = os.path.join(data_dir, f"{safe_name}_pre.tif")
    post_path = os.path.join(data_dir, f"{safe_name}_post.tif")
    if not (os.path.exists(pre_path) and os.path.exists(post_path)):
        return None
    pre_arr, transform, crs = load_geotiff(pre_path)
    post_arr, _, _ = load_geotiff(post_path)
    return pre_arr, post_arr, transform, crs


def real_data_available(data_dir=None):
    """Return dict of event_name -> bool indicating which events have real data."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
    result = {}
    for name in FLOOD_EVENTS:
        safe = name.replace(" ", "_")
        result[name] = (
            os.path.exists(os.path.join(data_dir, f"{safe}_pre.tif")) and
            os.path.exists(os.path.join(data_dir, f"{safe}_post.tif"))
        )
    return result


def detect_floods(pre_arr, post_arr, transform, filter_size=7):
    """
    Full flood detection pipeline.
    Returns dict with all outputs.
    """
    # 1. Apply Lee speckle filter
    pre_f = _lee_filter(pre_arr, size=filter_size)
    post_f = _lee_filter(post_arr, size=filter_size)

    # 2. Convert to dB
    eps = 1e-10
    pre_db = 10 * np.log10(pre_f + eps)
    post_db = 10 * np.log10(post_f + eps)

    # 3. Log-ratio change detection
    log_ratio = post_db - pre_db   # negative = backscatter decrease = flooding

    # 4. Otsu threshold on negative change
    change_neg = -log_ratio        # flip: positive values = potential flood
    try:
        otsu_thresh = threshold_otsu(change_neg)
    except Exception:
        otsu_thresh = np.percentile(change_neg, 75)

    raw_flood_mask = change_neg > otsu_thresh

    # 5. Morphological clean-up — remove noise, fill holes
    flood_mask = binary_opening(raw_flood_mask, disk(2))
    flood_mask = binary_closing(flood_mask, disk(3))

    # 6. Area calculation
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)
    # approximate metres per degree at equator
    m_per_deg_lon = 111_320
    m_per_deg_lat = 110_540
    pixel_area_km2 = (pixel_width * m_per_deg_lon / 1000) * (pixel_height * m_per_deg_lat / 1000)
    flooded_pixels = flood_mask.sum()
    flooded_area_km2 = flooded_pixels * pixel_area_km2

    # 7. Severity rating
    if flooded_area_km2 < 200:
        severity = "Minor"
        severity_color = "#22c55e"
    elif flooded_area_km2 < 2000:
        severity = "Moderate"
        severity_color = "#f59e0b"
    else:
        severity = "Severe"
        severity_color = "#ef4444"

    return {
        "pre_db": pre_db,
        "post_db": post_db,
        "log_ratio": log_ratio,
        "flood_mask": flood_mask,
        "flooded_area_km2": flooded_area_km2,
        "flooded_pixels": int(flooded_pixels),
        "total_pixels": int(flood_mask.size),
        "flood_fraction": float(flooded_pixels / flood_mask.size * 100),
        "otsu_thresh": float(otsu_thresh),
        "severity": severity,
        "severity_color": severity_color,
    }


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

def make_comparison_figure(pre_db, post_db, flood_mask, event_name, result, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0f172a")
    vmin, vmax = -25, 0

    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Pre-flood
    im0 = axes[0].imshow(pre_db, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Pre-Flood SAR (VV dB)\n{FLOOD_EVENTS[event_name]['date_pre']}",
                      color="#f1f5f9", fontsize=12, pad=10)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.02).ax.tick_params(colors="#94a3b8")

    # Post-flood
    im1 = axes[1].imshow(post_db, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Post-Flood SAR (VV dB)\n{FLOOD_EVENTS[event_name]['date_post']}",
                      color="#f1f5f9", fontsize=12, pad=10)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.04, pad=0.02).ax.tick_params(colors="#94a3b8")

    # Flood extent map
    axes[2].imshow(post_db, cmap="gray", vmin=vmin, vmax=vmax, alpha=0.6)
    flood_overlay = np.zeros((*flood_mask.shape, 4), dtype=np.float32)
    flood_overlay[flood_mask] = [0.22, 0.51, 0.94, 0.85]  # blue flood
    axes[2].imshow(flood_overlay)
    axes[2].set_title(
        f"Flood Extent Map\n{result['flooded_area_km2']:,.0f} km² — {result['severity']}",
        color="#f1f5f9", fontsize=12, pad=10
    )
    axes[2].axis("off")
    legend_elements = [
        Patch(facecolor="#3b82f6", alpha=0.85, label=f"Flooded ({result['flood_fraction']:.1f}% of scene)"),
        Patch(facecolor="#555", label="Land / Permanent water"),
    ]
    axes[2].legend(handles=legend_elements, loc="lower right",
                   facecolor="#1e293b", edgecolor="#334155",
                   labelcolor="#e2e8f0", fontsize=9)

    plt.suptitle(f"Flood Detection — {event_name}", color="#f1f5f9", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def make_change_figure(log_ratio, flood_mask, event_name, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Log-ratio heatmap
    lim = max(abs(np.percentile(log_ratio, 2)), abs(np.percentile(log_ratio, 98)))
    im = axes[0].imshow(log_ratio, cmap="RdBu", vmin=-lim, vmax=lim)
    axes[0].set_title("Log-Ratio Change (post − pre dB)\nBlue = backscatter decrease = flooding",
                      color="#f1f5f9", fontsize=11, pad=10)
    axes[0].axis("off")
    cb = plt.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors="#94a3b8")
    cb.set_label("dB change", color="#94a3b8")

    # Binary flood mask
    cmap_flood = mcolors.ListedColormap(["#1e293b", "#3b82f6"])
    axes[1].imshow(flood_mask.astype(int), cmap=cmap_flood, vmin=0, vmax=1)
    axes[1].set_title("Binary Flood Mask (Otsu threshold)\nBlue = detected flood pixels",
                      color="#f1f5f9", fontsize=11, pad=10)
    axes[1].axis("off")

    plt.suptitle(f"Change Detection Analysis — {event_name}",
                 color="#f1f5f9", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def make_stats_figure(results_all, save_path=None):
    """Bar chart comparing flooded area across all events."""
    events = list(results_all.keys())
    areas = [results_all[e]["flooded_area_km2"] for e in events]
    colors = [results_all[e]["severity_color"] for e in events]
    short = [e.split(" ")[0] + "\n" + " ".join(e.split(" ")[1:]) for e in events]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    bars = ax.bar(short, areas, color=colors, edgecolor="#0f172a", linewidth=1.5, width=0.6)
    for bar, area in zip(bars, areas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{area:,.0f} km²", ha="center", va="bottom",
                color="#e2e8f0", fontsize=10, fontweight="bold")

    ax.set_ylabel("Flooded Area (km²)", color="#94a3b8", fontsize=11)
    ax.set_title("Flood Extent Comparison Across Events",
                 color="#f1f5f9", fontsize=13, fontweight="bold", pad=14)
    ax.yaxis.label.set_color("#94a3b8")
    ax.tick_params(axis="x", colors="#cbd5e1")

    legend_elements = [
        Patch(facecolor="#22c55e", label="Minor (<200 km²)"),
        Patch(facecolor="#f59e0b", label="Moderate (200–2,000 km²)"),
        Patch(facecolor="#ef4444", label="Severe (>2,000 km²)"),
    ]
    ax.legend(handles=legend_elements, facecolor="#0f172a",
              edgecolor="#334155", labelcolor="#e2e8f0")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ─────────────────────────────────────────────
# OpenEO real data download (requires credentials)
# ─────────────────────────────────────────────

def download_sentinel1_openeo(event_name, output_dir, username=None, password=None):
    """
    Download real Sentinel-1 SAR data from Copernicus via OpenEO.
    Requires a free account at https://dataspace.copernicus.eu
    Returns paths to pre/post GeoTIFF files, or None if download fails.
    """
    try:
        import openeo
        event = FLOOD_EVENTS[event_name]
        bbox = event["bbox"]

        conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
        conn.authenticate_oidc_credentials(username=username, password=password)

        spatial_extent = {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]}

        def _download(date_start, date_end, out_path):
            cube = conn.load_collection(
                "SENTINEL1_GRD",
                spatial_extent=spatial_extent,
                temporal_extent=[date_start, date_end],
                bands=["VV"],
            )
            cube = cube.sar_backscatter(coefficient="sigma0-ellipsoid")
            cube.download(out_path, format="GTiff")

        pre_path = os.path.join(output_dir, f"{event_name.replace(' ', '_')}_pre.tif")
        post_path = os.path.join(output_dir, f"{event_name.replace(' ', '_')}_post.tif")

        _download(event["date_pre"],
                  str(int(event["date_pre"][:4])) + "-" + event["date_pre"][5:7] + "-28",
                  pre_path)
        _download(event["date_post"],
                  str(int(event["date_post"][:4])) + "-" + event["date_post"][5:7] + "-28",
                  post_path)

        return pre_path, post_path

    except Exception as e:
        print(f"OpenEO download failed: {e}")
        return None, None
