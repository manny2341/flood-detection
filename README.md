# 🌊 Flood Detection & Mapping System

A satellite-based flood detection system using Sentinel-1 SAR (Synthetic Aperture Radar) imagery and change detection algorithms. Detects flooded areas by comparing pre- and post-flood radar backscatter, maps flood extent, calculates affected area in km², and classifies severity — deployed as an interactive Streamlit web app.

## Demo

Select a real flood event → AI analyses SAR backscatter change → Flood extent map with area statistics and severity rating.

## Results

| Event | Country | Flooded Area | Severity |
|-------|---------|-------------|---------|
| Pakistan Monsoon | Pakistan | ~15,000 km² | 🔴 Severe |
| Storm Daniel | Libya | ~5,000 km² | 🔴 Severe |
| Annual Floods | Nigeria | ~10,000 km² | 🔴 Severe |
| Sylhet Floods | Bangladesh | ~3,800 km² | 🔴 Severe |
| Queensland/NSW | Australia | ~3,800 km² | 🔴 Severe |

## Features

- 5 real flood events — Pakistan 2022, Libya 2023, Nigeria 2022, Bangladesh 2022, Australia 2022
- Lee speckle filter — removes SAR multiplicative noise before analysis
- Log-ratio change detection — `post_dB − pre_dB` highlights backscatter decrease
- Otsu thresholding — automatic optimal threshold separating flood from land
- Morphological clean-up — binary opening + closing to remove noise and fill holes
- Flood extent map with blue overlay on SAR imagery
- Area calculation in km² using geographic pixel dimensions
- Severity classification: Minor (<200 km²) / Moderate (200–2,000 km²) / Severe (>2,000 km²)
- Upload your own GeoTIFF — works on any Sentinel-1 SAR file
- OpenEO integration — download real satellite data with Copernicus credentials
- Multi-event comparison chart

## Why SAR Over Optical?

| | SAR (Sentinel-1) | Optical (Sentinel-2) |
|---|---|---|
| Cloud penetration | ✅ Works through clouds | ❌ Blocked by clouds |
| Night operation | ✅ Active sensor — day or night | ❌ Needs sunlight |
| Water detection | ✅ Low backscatter = water | ✅ NDWI index |
| Flood timing | ✅ Captures ongoing floods | ⚠️ May miss cloud-covered events |

Floods happen during storms — clouds are always present. SAR is the correct sensor choice.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Satellite Data | Sentinel-1 SAR (VV polarisation) |
| Data Access | OpenEO + Copernicus Data Space |
| Speckle Filtering | Lee Filter (scipy.ndimage) |
| Thresholding | Otsu (scikit-image) |
| Morphology | Binary opening/closing (scikit-image) |
| Geospatial I/O | Rasterio |
| Web App | Streamlit |
| Visualisation | Matplotlib |
| Language | Python |

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/manny2341/flood-detection.git
cd flood-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the app**
```bash
streamlit run flood_app.py
```

Open `http://localhost:8501`

## How It Works

1. **Pre-flood SAR scene** loaded (VV band, linear scale)
2. **Post-flood SAR scene** loaded for same geographic area
3. **Lee speckle filter** applied to both scenes — reduces multiplicative SAR noise
4. Both scenes converted to **dB scale**: `10 × log₁₀(linear)`
5. **Log-ratio computed**: `change = post_dB − pre_dB`
   - Flooded land has lower backscatter than dry land → strong negative change
   - Permanent water unchanged → near-zero change
6. **Otsu threshold** applied to `-change` — automatically finds optimal cutoff
7. **Morphological operations** clean the binary mask (opening removes noise, closing fills holes)
8. **Area calculated** from pixel count × geographic pixel size
9. **Severity classified** by flooded area threshold

## Data Sources

- **Synthetic mode**: Realistic SAR scenes generated from statistical models (speckle + terrain + flood simulation) for immediate demo without credentials
- **OpenEO mode**: Real Sentinel-1 GRD data from [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) — free account required
- **Upload mode**: Your own Sentinel-1 GeoTIFF files from [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) (replaced scihub in 2023)

## Project Structure

```
flood-detection/
├── flood_app.py         # Streamlit web application
├── flood_engine.py      # Core pipeline: SAR generation, Lee filter, log-ratio, Otsu, visualisation
├── outputs/             # Generated flood maps and comparison charts
├── data/                # Downloaded GeoTIFF files (auto-created)
├── requirements.txt
└── README.md
```

## My Other Geospatial & Remote Sensing Projects

| Project | Description | Repo |
|---------|-------------|------|
| Wildfire Detection | YOLOv8 on Sentinel-2 satellite imagery — dNBR severity mapping | [wildfire-detection](https://github.com/manny2341/wildfire-detection-and-monitoring-) |
| Network Intrusion Detector | IoT traffic classification — C&C, DDoS, PortScan detection | [network-intrusion-detector](https://github.com/manny2341/network-intrusion-detector) |
| Crop Disease Detector | EfficientNetV2 — 15 plant diseases from leaf photos | [crop-disease-detector](https://github.com/manny2341/crop-disease-detector) |

## Author

[@manny2341](https://github.com/manny2341)
