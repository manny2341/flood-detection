"""
download_data.py — Download real Sentinel-1 SAR data from Microsoft Planetary Computer
Free access, no authentication required.

Usage:
    python3 download_data.py                    # download all 5 events
    python3 download_data.py "Pakistan 2022"    # download one event
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import planetary_computer
import pystac_client
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Event definitions (bbox, pre/post date ranges) ──────────────────────────
EVENTS = {
    "Pakistan 2022": {
        "bbox": [67.0, 27.0, 69.0, 29.0],
        "pre_start": "2022-06-01", "pre_end": "2022-07-15",
        "post_start": "2022-08-15", "post_end": "2022-09-30",
        "description": "Catastrophic monsoon floods — ~33M people affected",
    },
    "Libya 2023": {
        "bbox": [21.8, 32.2, 23.2, 33.2],
        "pre_start": "2023-07-01", "pre_end": "2023-09-09",
        "post_start": "2023-09-11", "post_end": "2023-10-01",
        "description": "Storm Daniel flash floods — Derna city destroyed",
    },
    "Nigeria 2022": {
        "bbox": [6.2, 6.2, 8.2, 8.2],
        "pre_start": "2022-07-01", "pre_end": "2022-09-01",
        "post_start": "2022-10-01", "post_end": "2022-11-15",
        "description": "Worst flooding in a decade — 1.4M displaced",
    },
    "Bangladesh 2022": {
        "bbox": [91.2, 24.2, 92.4, 25.2],
        "pre_start": "2022-04-01", "pre_end": "2022-05-15",
        "post_start": "2022-06-01", "post_end": "2022-07-15",
        "description": "Sylhet division flash floods — worst in 20 years",
    },
    "Australia 2022": {
        "bbox": [152.6, -27.8, 153.8, -26.6],
        "pre_start": "2021-12-01", "pre_end": "2022-01-15",
        "post_start": "2022-02-20", "post_end": "2022-03-15",
        "description": "Queensland / NSW — Brisbane inundated",
    },
}

TARGET_SIZE = 512   # pixels (square crop)


def connect_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def search_scenes(catalog, bbox, date_start, date_end, max_items=5):
    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        max_items=max_items,
    )
    return list(search.items())


def download_vv_tile(item, bbox, out_path, target_size=TARGET_SIZE):
    """Download VV band clipped to bbox, resampled to target_size × target_size."""
    vv_url = item.assets["vv"].href

    with rasterio.open(vv_url) as src:
        # Convert bbox (lon/lat) to native CRS
        from rasterio.warp import transform_bounds
        native_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)

        window = from_bounds(*native_bbox, transform=src.transform)
        # Clip window to valid data extent
        window = rasterio.windows.intersection(
            window,
            rasterio.windows.Window(0, 0, src.width, src.height)
        )

        if window.width <= 0 or window.height <= 0:
            return None

        # Read and resample to target_size
        data = src.read(
            1,
            window=window,
            out_shape=(target_size, target_size),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        # Build output transform
        from rasterio.transform import from_bounds as tfb
        out_transform = tfb(*bbox, target_size, target_size)
        out_crs = rasterio.crs.CRS.from_epsg(4326)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=target_size, width=target_size,
        count=1, dtype="float32",
        crs=out_crs, transform=out_transform,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)

    return out_path


def download_event(event_name, catalog=None, force=False):
    if catalog is None:
        catalog = connect_catalog()

    ev = EVENTS[event_name]
    safe_name = event_name.replace(" ", "_")
    pre_path = os.path.join(DATA_DIR, f"{safe_name}_pre.tif")
    post_path = os.path.join(DATA_DIR, f"{safe_name}_post.tif")

    if os.path.exists(pre_path) and os.path.exists(post_path) and not force:
        print(f"  [{event_name}] Already downloaded — skipping")
        return pre_path, post_path

    print(f"  [{event_name}] Searching pre-flood scenes ({ev['pre_start']} → {ev['pre_end']})...")
    pre_items = search_scenes(catalog, ev["bbox"], ev["pre_start"], ev["pre_end"])
    if not pre_items:
        print(f"  [{event_name}] No pre-flood scenes found")
        return None, None

    print(f"  [{event_name}] Searching post-flood scenes ({ev['post_start']} → {ev['post_end']})...")
    post_items = search_scenes(catalog, ev["bbox"], ev["post_start"], ev["post_end"])
    if not post_items:
        print(f"  [{event_name}] No post-flood scenes found")
        return None, None

    print(f"  [{event_name}] Downloading pre-flood tile ({pre_items[0].id})...")
    result = download_vv_tile(pre_items[0], ev["bbox"], pre_path)
    if not result:
        print(f"  [{event_name}] Pre-flood tile out of scene bounds")
        return None, None

    print(f"  [{event_name}] Downloading post-flood tile ({post_items[0].id})...")
    result = download_vv_tile(post_items[0], ev["bbox"], post_path)
    if not result:
        print(f"  [{event_name}] Post-flood tile out of scene bounds")
        return None, None

    # Quick sanity check
    import rasterio as rio
    with rio.open(pre_path) as src:
        pre_data = src.read(1)
    with rio.open(post_path) as src:
        post_data = src.read(1)

    print(f"  [{event_name}] ✓ Pre: {pre_data.shape} | range [{pre_data.min():.4f}, {pre_data.max():.4f}]")
    print(f"  [{event_name}] ✓ Post: {post_data.shape} | range [{post_data.min():.4f}, {post_data.max():.4f}]")
    return pre_path, post_path


def download_all(force=False):
    print("Connecting to Microsoft Planetary Computer...")
    catalog = connect_catalog()
    print(f"Connected. Downloading Sentinel-1 SAR data for {len(EVENTS)} events.\n")

    results = {}
    for name in EVENTS:
        try:
            pre, post = download_event(name, catalog, force=force)
            results[name] = {"pre": pre, "post": post, "ok": pre is not None}
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            results[name] = {"pre": None, "post": None, "ok": False, "error": str(e)}
        print()

    print("── Summary ──────────────────────────────")
    for name, r in results.items():
        status = "✓" if r["ok"] else "✗"
        print(f"  {status} {name}")
    print()
    ok = sum(1 for r in results.values() if r["ok"])
    print(f"Downloaded {ok}/{len(EVENTS)} events to {DATA_DIR}/")
    return results


if __name__ == "__main__":
    force = "--force" in sys.argv
    target = next((a for a in sys.argv[1:] if not a.startswith("--")), None)

    os.makedirs(DATA_DIR, exist_ok=True)

    if target:
        if target not in EVENTS:
            print(f"Unknown event: '{target}'")
            print(f"Available: {', '.join(EVENTS.keys())}")
            sys.exit(1)
        catalog = connect_catalog()
        download_event(target, catalog, force=force)
    else:
        download_all(force=force)
