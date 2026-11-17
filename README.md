# SkyTrace — Acoustic Aircraft Doppler Geolocation (`aerodoppler`)

An open-source implementation and interactive OSINT workstation reproducing the acoustic aircraft Doppler geolocation pipeline demonstrated in Colsto's video *"I discovered a sound that leaks your exact location (and it's everywhere)"* (`UytYrhYLOKI`).

> **Educational & Defensive Research Tool:** Demonstrates how sub-audible background aircraft engine/propeller Doppler curves in audio or video recordings correlate with public ADS-B flight data and terrain elevation (DEM) to estimate recording coordinates, and tests why naive audio countermeasures fail.

---

## How the Pipeline Works

1. **High-Resolution Whitened Spectrogram (`aerodoppler/spectrogram.py`)**
   - Computes long-window STFT (e.g., 4s Hann window, 2x zero-padding) zoomed into low-frequency propeller fundamentals & harmonics (`40–360 Hz`).
   - Applies two-way robust noise equalization (per-bin stationary hum removal + per-frame transient removal + MAD Z-score whitening) to expose faint, human-inaudible Doppler tracks.
   - Automatically detects flyover peaks and fits an ideal constant-velocity Doppler curve to extract conservative bounds on CPA time, $f_0$, speed $v$, and slant distance $d_{\text{CPA}}$.

2. **ADS-B Ingestion & ICAO Aircraft Type Filtering (`aerodoppler/adsb.py`, `aerodoppler/aircraft_types.py`)**
   - Supports `adsb.lol` / `readsb` `.json` and `.json.gz` trace files, CSV, Parquet, unified JSON, and live `adsb.lol` v2 API queries.
   - Implements the ICAO 8643 jet-exclusion strategy (`exclude_jets`): removes known commercial/business jets while retaining propeller, turboprop, helicopter, and unmapped aircraft types.

3. **3D Spatial Grid, DEM Elevation & Sound Propagation Delay (`aerodoppler/dem.py`, `aerodoppler/physics.py`)**
   - Builds a 200m local ENU spatial lattice around candidate flight corridors and enriches each grid cell with terrain elevation (MSL) via GeoTIFF/ASCII/NPZ `GridDEM` or `OpenMeteoDEM`.
   - Computes 3D ECEF slant distance, closest point of approach ($t_{\text{CPA}}$), and **delayed CPA time**:
     $$t_{\text{CPA,delayed}} = t_{\text{CPA}} + \frac{d_{\text{CPA}}}{c}$$

4. **Time-Offset & Silence Constraint Filtering (`aerodoppler/locator.py`)**
   - Filters grid points whose delayed CPA separation matches the observed inter-event gap (e.g., 920s ± tolerance).
   - Applies the **Silence Constraint Filter**: disqualifies points (such as busy airport corridors) where any other aircraft would have been heard within the audible radius (e.g., 10 km) during the quiet interval between flyovers.

5. **Retarded-Time 3D Doppler Modeling & Locked Scoring (`aerodoppler/physics.py`, `aerodoppler/locator.py`)**
   - Refines surviving 200m cells into a **100m grid** (and **25m** fine heatmap around the top winner).
   - Evaluates the aircraft's radial velocity at **retarded emission time** $t_e$, found by inverting:
     $$t_h = t_e + \frac{\|\vec{p}_{\text{plane}}(t_e) - \vec{p}_{\text{obs}}\|}{c}, \qquad f(t_h) = f_0 \frac{c}{c - v_r(t_e)}$$
   - **Two-Event Locked Scoring:** Sweeps measurement-error offsets ($\delta t_1, \delta f_{0,1}$) on Event 1 while keeping Event 2's relative time placement strictly locked to the physics-derived $\Delta t_{\text{delayed}}$.
   - **Single-Event Traced/UTC Scoring:** Supports 1-minute clips with known UTC time and optional user-traced curve points.

6. **Countermeasures & Privacy Lab (`aerodoppler/countermeasures.py`)**
   - Tests high-pass filtering + lossy compression + post-whitening recovery, speech/broadband masking, steep 450 Hz band-stop (destroying voice clarity), and time-cut splicing.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Interactive Web Workstation
```bash
python -m aerodoppler.cli serve --port 8080
```
Open `http://127.0.0.1:8080` in your browser. A built-in synthetic benchmark scenario loads automatically so you can inspect the whitened spectrogram, tune curve fits, run the 200m -> 100m -> 25m geolocation funnel, view the Event 1 vs Event 2 scatter plot & Leaflet map, compare **Retarded vs Naive** curves, and test countermeasures.

### 3. Run CLI Benchmarks or Real Data
```bash
# Built-in 90s offset benchmark
python -m aerodoppler.cli demo --preset quick_two_event

# Built-in Colsto 920s offset benchmark (17-minute audio)
python -m aerodoppler.cli demo --preset colsto_920s

# Built-in 1-minute faint single-flight benchmark
python -m aerodoppler.cli demo --preset single_1min_faint

# Locate your own audio/video file against ADS-B traces
python -m aerodoppler.cli locate \
  --audio recording.wav \
  --adsb ./traces/ \
  --mode two_event \
  --dem open-meteo \
  --auto-detect
```

### 4. Run Tests
```bash
python -m unittest discover -s tests -v
```
