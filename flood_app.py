"""
flood_app.py — Streamlit web app for Flood Detection & Mapping
Run: streamlit run flood_app.py
"""

import os
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import rasterio
import tempfile

from flood_engine import (
    FLOOD_EVENTS,
    generate_sar_scene,
    load_geotiff,
    load_real_data,
    real_data_available,
    save_geotiff,
    detect_floods,
    make_comparison_figure,
    make_change_figure,
    make_stats_figure,
    download_sentinel1_openeo,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Detection System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .main .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #f1f5f9; }
    .metric-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }
    .severity-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
    }
    .stSelectbox label, .stFileUploader label { color: #94a3b8 !important; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }
    .stButton > button:hover { background: #2563eb; }
    div[data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Flood Detection")
    st.markdown("*Sentinel-1 SAR · Change Detection*")
    st.markdown("---")

    mode = st.radio(
        "Data Source",
        ["Preset Events (Synthetic SAR)", "Upload GeoTIFF", "Download via OpenEO"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Algorithm**")
    st.markdown("""
- Lee speckle filter
- Log-ratio change detection
- Otsu thresholding
- Morphological clean-up
    """)

    st.markdown("---")
    st.markdown("**Real Events Included**")
    for name, ev in FLOOD_EVENTS.items():
        sev = ev["severity"]
        col = "🔴" if sev == "Severe" else "🟡"
        st.markdown(f"{col} {name}")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("# 🌊 Flood Detection & Mapping System")
st.markdown("*Sentinel-1 SAR change detection — log-ratio analysis + Otsu thresholding on real flood events*")
st.markdown("---")


# ─────────────────────────────────────────────
# MODE 1: Preset events
# ─────────────────────────────────────────────
if mode == "Preset Events (Synthetic SAR)":

    available = real_data_available()
    total_real = sum(available.values())

    col1, col2 = st.columns([2, 1])
    with col1:
        event_name = st.selectbox("Select Flood Event", list(FLOOD_EVENTS.keys()))
    with col2:
        run_all = st.checkbox("Analyse all 5 events", value=False)

    # Data source badge
    has_real = available.get(event_name, False)
    if has_real:
        st.success(f"✅ Real Sentinel-1 SAR data available for **{event_name}** — downloaded from Microsoft Planetary Computer")
    else:
        st.info("ℹ️ Using synthetic SAR scene. Run `python3 download_data.py` to download real data.")

    if total_real > 0:
        st.caption(f"Real satellite data available for {total_real}/5 events")

    event = FLOOD_EVENTS[event_name]
    st.markdown(f"""
    > **{event_name}** &nbsp;|&nbsp; {event['date_pre']} → {event['date_post']}
    > {event['description']}
    """)

    if st.button("Run Flood Detection", key="run_single"):
        with st.spinner("Loading SAR data and running detection..."):

            # Use real data if available, else synthetic
            real = load_real_data(event_name)
            if real is not None:
                pre_arr, post_arr, transform, crs = real
                data_source = "🛰️ Real Sentinel-1 GRD (Microsoft Planetary Computer)"
            else:
                pre_arr, transform, crs = generate_sar_scene(event_name, "pre", seed=42)
                post_arr, _, _ = generate_sar_scene(event_name, "post", seed=99)
                data_source = "🧪 Synthetic SAR scene"

            # Run detection
            result = detect_floods(pre_arr, post_arr, transform)

            # Figures
            fig_comp = make_comparison_figure(
                result["pre_db"], result["post_db"],
                result["flood_mask"], event_name, result
            )
            fig_change = make_change_figure(
                result["log_ratio"], result["flood_mask"], event_name
            )

        # ── Metrics row ──
        st.markdown("### Results")
        st.caption(f"Data source: {data_source}")
        m1, m2, m3, m4, m5 = st.columns(5)
        sev_colors = {"Minor": "#22c55e", "Moderate": "#f59e0b", "Severe": "#ef4444"}
        sc = sev_colors[result["severity"]]

        with m1:
            st.metric("Flooded Area", f"{result['flooded_area_km2']:,.0f} km²")
        with m2:
            st.metric("Flood Coverage", f"{result['flood_fraction']:.1f}%")
        with m3:
            st.metric("Flooded Pixels", f"{result['flooded_pixels']:,}")
        with m4:
            st.metric("Otsu Threshold", f"{result['otsu_thresh']:.2f} dB")
        with m5:
            st.markdown(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px;text-align:center">
            <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em">Severity</div>
            <div style="font-size:1.5rem;font-weight:700;color:{sc};margin-top:4px">{result['severity']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Comparison figure ──
        st.markdown("### SAR Imagery — Before / After / Flood Extent")
        st.pyplot(fig_comp, use_container_width=True)
        plt.close(fig_comp)

        st.markdown("### Change Detection Analysis")
        st.pyplot(fig_change, use_container_width=True)
        plt.close(fig_change)

        # ── Bbox info ──
        bbox = event["bbox"]
        st.markdown("### Scene Coverage")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            | Field | Value |
            |-------|-------|
            | Country | {event['country']} |
            | Pre-flood date | {event['date_pre']} |
            | Post-flood date | {event['date_post']} |
            | Bounding box | {bbox[0]}°E, {bbox[1]}°N → {bbox[2]}°E, {bbox[3]}°N |
            """)
        with c2:
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Scene size | 512 × 512 px |
            | Flooded area | {result['flooded_area_km2']:,.0f} km² |
            | Flood fraction | {result['flood_fraction']:.1f}% of scene |
            | Severity | {result['severity']} |
            """)

    # ── All events comparison ──
    if run_all:
        st.markdown("---")
        st.markdown("### All Events — Comparative Analysis")
        if st.button("Analyse All 5 Events", key="run_all"):
            results_all = {}
            progress = st.progress(0)
            status = st.empty()
            for i, (name, _) in enumerate(FLOOD_EVENTS.items()):
                real = load_real_data(name)
                if real is not None:
                    status.text(f"Processing {name} (real Sentinel-1 data)...")
                    pre_arr, post_arr, transform, crs = real
                else:
                    status.text(f"Processing {name} (synthetic)...")
                    pre_arr, transform, crs = generate_sar_scene(name, "pre", seed=42)
                    post_arr, _, _ = generate_sar_scene(name, "post", seed=99)
                results_all[name] = detect_floods(pre_arr, post_arr, transform)
                progress.progress((i + 1) / len(FLOOD_EVENTS))

            status.text("Generating comparison chart...")
            fig_stats = make_stats_figure(results_all)
            st.pyplot(fig_stats, use_container_width=True)
            plt.close(fig_stats)

            # Summary table
            st.markdown("### Summary Table")
            rows = []
            for name, r in results_all.items():
                ev = FLOOD_EVENTS[name]
                rows.append({
                    "Event": name,
                    "Country": ev["country"],
                    "Flooded Area (km²)": f"{r['flooded_area_km2']:,.0f}",
                    "Coverage (%)": f"{r['flood_fraction']:.1f}%",
                    "Severity": r["severity"],
                })
            import pandas as pd
            st.dataframe(
                pd.DataFrame(rows).set_index("Event"),
                use_container_width=True,
            )
            progress.empty()
            status.empty()


# ─────────────────────────────────────────────
# MODE 2: Upload GeoTIFF
# ─────────────────────────────────────────────
elif mode == "Upload GeoTIFF":
    st.markdown("### Upload Sentinel-1 SAR GeoTIFF Files")
    st.info("Upload pre-flood and post-flood SAR GeoTIFF files (VV band, linear scale). "
            "You can download Sentinel-1 data free from [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) — the replacement for the old scihub (shut down May 2023).")

    col1, col2 = st.columns(2)
    with col1:
        pre_file = st.file_uploader("Pre-Flood SAR (.tif)", type=["tif", "tiff"])
    with col2:
        post_file = st.file_uploader("Post-Flood SAR (.tif)", type=["tif", "tiff"])

    event_label = st.text_input("Event Label (for chart titles)", "Custom Event")

    if pre_file and post_file and st.button("Run Detection on Uploaded Files"):
        with st.spinner("Processing uploaded files..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                pre_path = os.path.join(tmpdir, "pre.tif")
                post_path = os.path.join(tmpdir, "post.tif")
                with open(pre_path, "wb") as f:
                    f.write(pre_file.read())
                with open(post_path, "wb") as f:
                    f.write(post_file.read())

                pre_arr, transform, crs = load_geotiff(pre_path)
                post_arr, _, _ = load_geotiff(post_path)

                # Resize if arrays are different shapes
                if pre_arr.shape != post_arr.shape:
                    from skimage.transform import resize
                    post_arr = resize(post_arr, pre_arr.shape, preserve_range=True).astype(np.float32)

                result = detect_floods(pre_arr, post_arr, transform)

                fig_comp = make_comparison_figure(
                    result["pre_db"], result["post_db"],
                    result["flood_mask"], event_label, result
                )
                fig_change = make_change_figure(
                    result["log_ratio"], result["flood_mask"], event_label
                )

        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Flooded Area", f"{result['flooded_area_km2']:,.0f} km²")
        with m2: st.metric("Flood Coverage", f"{result['flood_fraction']:.1f}%")
        with m3: st.metric("Flooded Pixels", f"{result['flooded_pixels']:,}")
        with m4: st.metric("Severity", result["severity"])

        st.pyplot(fig_comp, use_container_width=True)
        plt.close(fig_comp)
        st.pyplot(fig_change, use_container_width=True)
        plt.close(fig_change)


# ─────────────────────────────────────────────
# MODE 3: OpenEO download
# ─────────────────────────────────────────────
elif mode == "Download via OpenEO":
    st.markdown("### Download Real Sentinel-1 Data via OpenEO")
    st.info("""
    Download real Sentinel-1 SAR data directly from the Copernicus Data Space Ecosystem.
    **Free account required** — register at [dataspace.copernicus.eu](https://dataspace.copernicus.eu)
    """)

    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Copernicus Username")
    with col2:
        password = st.text_input("Copernicus Password", type="password")

    event_name = st.selectbox("Select Flood Event", list(FLOOD_EVENTS.keys()), key="eo_event")

    if st.button("Download & Analyse"):
        if not username or not password:
            st.error("Please enter your Copernicus credentials.")
        else:
            with st.spinner("Downloading Sentinel-1 data from Copernicus (this may take a few minutes)..."):
                out_dir = os.path.join(os.path.expanduser("~"), "flood-detection", "data")
                pre_path, post_path = download_sentinel1_openeo(
                    event_name, out_dir, username, password
                )

            if pre_path and post_path:
                st.success("Download complete. Running detection...")
                pre_arr, transform, crs = load_geotiff(pre_path)
                post_arr, _, _ = load_geotiff(post_path)
                result = detect_floods(pre_arr, post_arr, transform)

                fig_comp = make_comparison_figure(
                    result["pre_db"], result["post_db"],
                    result["flood_mask"], event_name, result
                )
                st.pyplot(fig_comp, use_container_width=True)
                plt.close(fig_comp)
            else:
                st.error("Download failed. Check credentials or try the preset events mode.")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center;color:#475569;font-size:0.82rem">
Flood Detection System · Sentinel-1 SAR · Log-Ratio Change Detection · Otsu Thresholding ·
<a href="https://github.com/manny2341" style="color:#3b82f6">@manny2341</a>
</p>
""", unsafe_allow_html=True)
