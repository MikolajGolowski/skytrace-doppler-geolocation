"""FastAPI web server for SkyTrace / AeroDoppler interactive OSINT workstation."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aerodoppler.adsb import FlightTrack, fetch_adsblol_live, load_adsb_path
from aerodoppler.countermeasures import evaluate_countermeasure
from aerodoppler.dem import DEMProvider, FlatDEM, OpenMeteoDEM, SyntheticTerrainDEM
from aerodoppler.locator import EventQuery, run_geolocation_pipeline
from aerodoppler.physics import DEFAULT_SPEED_OF_SOUND, haversine_m, ideal_doppler_curve
from aerodoppler.simulator import DemoScenario, generate_demo_scenario
from aerodoppler.spectrogram import (
    SpectrogramData,
    boost_band_audio_wav_bytes,
    compute_spectrogram,
    detect_doppler_events,
    fit_ideal_doppler,
    load_audio,
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="SkyTrace / AeroDoppler", version="0.1.0")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class WorkspaceState:
    def __init__(self) -> None:
        self.samples: np.ndarray | None = None
        self.sr: int = 2000
        self.spec: SpectrogramData | None = None
        self.tracks: list[FlightTrack] = []
        self.dem: DEMProvider = FlatDEM(0.0)
        self.demo: DemoScenario | None = None
        self.source_label: str = "None"


STATE = WorkspaceState()


def _serialize_spectrogram(
    spec: SpectrogramData,
    t_range: tuple[float, float] | None = None,
    f_range: tuple[float, float] | None = None,
    max_w: int = 560,
    max_h: int = 240,
    use_whitened: bool = True,
) -> dict[str, Any]:
    """Downsample and uint8-quantize a spectrogram slice for fast Canvas rendering."""
    mat = spec.whitened if use_whitened else spec.raw_db
    t_mask = np.ones(len(spec.times), dtype=bool)
    f_mask = np.ones(len(spec.freqs), dtype=bool)
    if t_range:
        t_mask = (spec.times >= t_range[0]) & (spec.times <= t_range[1])
    if f_range:
        f_mask = (spec.freqs >= f_range[0]) & (spec.freqs <= f_range[1])

    times_sub = spec.times[t_mask]
    freqs_sub = spec.freqs[f_mask]
    sub = mat[np.ix_(f_mask, t_mask)]
    if sub.size == 0:
        return {"width": 0, "height": 0, "t_min": 0, "t_max": 1, "f_min": 40, "f_max": 360, "data_b64": ""}

    # Stride / max-pool downsample so bright thin curves never vanish on zoom-out
    step_t = max(1, int(np.ceil(sub.shape[1] / max_w)))
    step_f = max(1, int(np.ceil(sub.shape[0] / max_h)))

    if step_t > 1 or step_f > 1:
        trim_f = (sub.shape[0] // step_f) * step_f
        trim_t = (sub.shape[1] // step_t) * step_t
        sub_c = sub[:trim_f, :trim_t]
        sub_ds = sub_c.reshape(trim_f // step_f, step_f, trim_t // step_t, step_t).max(axis=(1, 3))
        times_ds = times_sub[:trim_t:step_t]
        freqs_ds = freqs_sub[:trim_f:step_f]
    else:
        sub_ds = sub
        times_ds = times_sub
        freqs_ds = freqs_sub

    lo = float(np.percentile(sub_ds, 2))
    hi = float(np.percentile(sub_ds, 99.5))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    norm = np.clip((sub_ds - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    # Flip vertically so row 0 is top (highest freq) for direct canvas ImageData
    u8 = np.flipud(u8)
    b64 = base64.b64encode(u8.tobytes(order="C")).decode("ascii")
    return {
        "width": int(u8.shape[1]),
        "height": int(u8.shape[0]),
        "t_min": round(float(times_ds[0]), 2),
        "t_max": round(float(times_ds[-1]), 2),
        "f_min": round(float(freqs_ds[0]), 2),
        "f_max": round(float(freqs_ds[-1]), 2),
        "duration_s": round(spec.duration_s, 2),
        "data_b64": b64,
    }


class DemoLoadReq(BaseModel):
    preset: Literal["quick_two_event", "colsto_920s", "single_1min_faint"] = "quick_two_event"


class SpecRecomputeReq(BaseModel):
    fmin: float = 40.0
    fmax: float = 360.0
    win_sec: float = 4.0
    hop_sec: float = 0.5
    t_min: float | None = None
    t_max: float | None = None
    use_whitened: bool = True


class FitEventReq(BaseModel):
    t_center_s: float
    f0_min: float = 65.0
    f0_max: float = 240.0


class IdealCurveReq(BaseModel):
    t_cpa_s: float
    f0_hz: float
    speed_ms: float
    d_cpa_m: float
    span_s: float = 45.0


class EventQueryPayload(BaseModel):
    t_cpa_s: float
    f0_hz: float
    speed_min_ms: float
    speed_max_ms: float
    d_min_m: float
    d_max_m: float
    t_tolerance_s: float = 6.0
    f0_tolerance_hz: float = 4.0
    traced_points: list[list[float]] = []


class LocateReq(BaseModel):
    mode: Literal["two_event", "single_event"] = "two_event"
    event1: EventQueryPayload
    event2: EventQueryPayload | None = None
    aircraft_filter_mode: Literal["exclude_jets", "prop_only", "all"] = "exclude_jets"
    dem_mode: Literal["auto", "flat", "open_meteo"] = "auto"
    enable_silence_filter: bool = True
    audible_radius_m: float = 10000.0
    known_utc_cpa1: float | None = None
    known_utc_tolerance_s: float = 35.0


class LiveAdsbReq(BaseModel):
    lat: float
    lon: float
    dist_nm: int = 25


class CountermeasureReq(BaseModel):
    t_cpa_s: float
    f0_hz: float
    speed_ms: float
    d_cpa_m: float
    method: Literal["highpass_boost", "steep_bandstop", "speech_masking", "time_cut"] = "highpass_boost"
    param: float = 160.0


@app.get("/")
def serve_index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return FileResponse(str(index_path))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "has_audio": STATE.samples is not None,
        "track_count": len(STATE.tracks),
        "source": STATE.source_label,
    }


@app.post("/api/demo/load")
def load_demo(req: DemoLoadReq) -> dict[str, Any]:
    sc = generate_demo_scenario(preset=req.preset)
    STATE.demo = sc
    STATE.samples = sc.samples
    STATE.sr = sc.sample_rate
    STATE.tracks = sc.tracks
    STATE.dem = sc.dem
    STATE.source_label = sc.name
    STATE.spec = compute_spectrogram(sc.samples, sc.sample_rate)

    max_ev = 2 if sc.mode == "two_event" else 1
    events = detect_doppler_events(STATE.spec, max_events=max_ev)

    return {
        "name": sc.name,
        "mode": sc.mode,
        "duration_s": round(STATE.spec.duration_s, 2),
        "track_count": len(sc.tracks),
        "known_utc_cpa1": sc.known_utc_cpa1,
        "recording_start_utc": sc.recording_start_utc,
        "spectrogram": _serialize_spectrogram(STATE.spec),
        "detected_events": [e.to_dict() for e in events],
        "ground_truth": {
            "lat": sc.true_lat,
            "lon": sc.true_lon,
            "elev_m": round(sc.true_elev_m, 1),
            "event1_audio_s": round(sc.true_event1_audio_s, 2),
            "event2_audio_s": round(sc.true_event2_audio_s, 2) if sc.true_event2_audio_s else None,
            "f0_1": sc.true_f0_1,
            "f0_2": sc.true_f0_2,
        },
        "tracks_preview": [
            {
                "icao24": t.icao24,
                "callsign": t.callsign,
                "type_code": t.type_code,
                "propulsion_class": t.propulsion_class,
                "median_speed_ms": round(t.median_speed_ms, 1),
            }
            for t in sc.tracks
        ],
    }


@app.post("/api/audio/upload")
async def upload_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        samples, sr = load_audio(tmp_path, target_sr=2000)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    STATE.demo = None
    STATE.samples = samples
    STATE.sr = sr
    STATE.source_label = file.filename or "Uploaded audio"
    STATE.spec = compute_spectrogram(samples, sr)
    events = detect_doppler_events(STATE.spec, max_events=3)
    return {
        "filename": STATE.source_label,
        "duration_s": round(STATE.spec.duration_s, 2),
        "spectrogram": _serialize_spectrogram(STATE.spec),
        "detected_events": [e.to_dict() for e in events],
    }


@app.get("/api/audio/boosted")
def get_boosted_audio(low_hz: float = 55.0, high_hz: float = 340.0) -> Response:
    if STATE.samples is None:
        raise HTTPException(status_code=400, detail="No audio loaded.")
    wav_bytes = boost_band_audio_wav_bytes(STATE.samples, STATE.sr, low_hz=low_hz, high_hz=high_hz)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/api/spectrogram/recompute")
def recompute_spec(req: SpecRecomputeReq) -> dict[str, Any]:
    if STATE.samples is None:
        raise HTTPException(status_code=400, detail="No audio loaded.")
    STATE.spec = compute_spectrogram(
        STATE.samples,
        STATE.sr,
        fmin=req.fmin,
        fmax=req.fmax,
        win_sec=req.win_sec,
        hop_sec=req.hop_sec,
    )
    t_range = (req.t_min, req.t_max) if req.t_min is not None and req.t_max is not None else None
    return {
        "spectrogram": _serialize_spectrogram(
            STATE.spec,
            t_range=t_range,
            f_range=(req.fmin, req.fmax),
            use_whitened=req.use_whitened,
        )
    }


@app.post("/api/events/fit")
def fit_event_at_time(req: FitEventReq) -> dict[str, Any]:
    if STATE.spec is None:
        raise HTTPException(status_code=400, detail="No spectrogram available.")
    ev = fit_ideal_doppler(
        STATE.spec,
        t_center_hint=req.t_center_s,
        f0_range=(req.f0_min, req.f0_max),
    )
    return {"event": ev.to_dict()}


@app.post("/api/events/ideal_curve")
def eval_ideal_curve(req: IdealCurveReq) -> dict[str, Any]:
    t_arr = np.linspace(req.t_cpa_s - req.span_s, req.t_cpa_s + req.span_s, 140)
    f_arr = ideal_doppler_curve(
        t_arr,
        f0=req.f0_hz,
        speed_ms=req.speed_ms,
        d_cpa_m=req.d_cpa_m,
        t_cpa_delayed=req.t_cpa_s,
        c=DEFAULT_SPEED_OF_SOUND,
    )
    return {
        "t": [round(float(x), 2) for x in t_arr],
        "f": [round(float(y), 2) for y in f_arr],
    }


@app.post("/api/adsb/upload")
async def upload_adsb(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = "".join(Path(file.filename or "tracks.json").suffixes) or ".json"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        tracks = load_adsb_path(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    STATE.tracks = tracks
    STATE.demo = None
    return {
        "track_count": len(tracks),
        "tracks_preview": [
            {
                "icao24": t.icao24,
                "callsign": t.callsign,
                "type_code": t.type_code,
                "propulsion_class": t.propulsion_class,
                "median_speed_ms": round(t.median_speed_ms, 1),
            }
            for t in tracks[:100]
        ],
    }


@app.post("/api/adsb/fetch_live")
def fetch_live_adsb(req: LiveAdsbReq) -> dict[str, Any]:
    try:
        ac = fetch_adsblol_live(req.lat, req.lon, req.dist_nm)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"adsb.lol query failed: {exc}") from exc
    return {"count": len(ac), "aircraft": ac[:80]}


@app.post("/api/locate")
def run_locate(req: LocateReq) -> dict[str, Any]:
    if STATE.spec is None:
        raise HTTPException(status_code=400, detail="Load audio or a demo scenario first.")
    if not STATE.tracks:
        raise HTTPException(status_code=400, detail="Load ADS-B tracks or a demo scenario first.")

    dem: DEMProvider
    if req.dem_mode == "open_meteo":
        dem = OpenMeteoDEM()
    elif req.dem_mode == "flat":
        dem = FlatDEM(0.0)
    else:
        dem = STATE.dem if STATE.demo is not None else FlatDEM(0.0)

    q1 = EventQuery(
        t_cpa_s=req.event1.t_cpa_s,
        f0_hz=req.event1.f0_hz,
        speed_min_ms=req.event1.speed_min_ms,
        speed_max_ms=req.event1.speed_max_ms,
        d_min_m=req.event1.d_min_m,
        d_max_m=req.event1.d_max_m,
        t_tolerance_s=req.event1.t_tolerance_s,
        f0_tolerance_hz=req.event1.f0_tolerance_hz,
        traced_points=[(float(p[0]), float(p[1])) for p in req.event1.traced_points],
    )
    q2 = None
    if req.mode == "two_event":
        if req.event2 is None:
            raise HTTPException(status_code=400, detail="two_event mode requires event2 parameters.")
        q2 = EventQuery(
            t_cpa_s=req.event2.t_cpa_s,
            f0_hz=req.event2.f0_hz,
            speed_min_ms=req.event2.speed_min_ms,
            speed_max_ms=req.event2.speed_max_ms,
            d_min_m=req.event2.d_min_m,
            d_max_m=req.event2.d_max_m,
            t_tolerance_s=req.event2.t_tolerance_s,
            f0_tolerance_hz=req.event2.f0_tolerance_hz,
        )

    report = run_geolocation_pipeline(
        tracks=STATE.tracks,
        spec=STATE.spec,
        event1=q1,
        event2=q2,
        mode=req.mode,
        dem=dem,
        aircraft_filter_mode=req.aircraft_filter_mode,
        known_utc_cpa1=req.known_utc_cpa1,
        known_utc_tolerance_s=req.known_utc_tolerance_s,
        audible_radius_m=req.audible_radius_m,
        enable_silence_filter=req.enable_silence_filter,
    )

    payload = report.to_dict()
    # Attach zoomed spectrogram patches for top-candidate overlay verification
    spec_slices = {}
    spec_slices["event1"] = _serialize_spectrogram(
        STATE.spec,
        t_range=(max(0.0, q1.t_cpa_s - 38.0), min(STATE.spec.duration_s, q1.t_cpa_s + 38.0)),
        f_range=(max(35.0, q1.f0_hz * 0.65), min(360.0, q1.f0_hz * 1.45)),
        max_w=360,
        max_h=180,
    )
    if q2 is not None:
        spec_slices["event2"] = _serialize_spectrogram(
            STATE.spec,
            t_range=(max(0.0, q2.t_cpa_s - 38.0), min(STATE.spec.duration_s, q2.t_cpa_s + 38.0)),
            f_range=(max(35.0, q2.f0_hz * 0.65), min(360.0, q2.f0_hz * 1.45)),
            max_w=360,
            max_h=180,
        )
    payload["spec_slices"] = spec_slices

    if STATE.demo is not None and report.top_candidates:
        top = report.top_candidates[0]
        payload["ground_truth_eval"] = {
            "true_lat": STATE.demo.true_lat,
            "true_lon": STATE.demo.true_lon,
            "error_m": round(haversine_m(STATE.demo.true_lat, STATE.demo.true_lon, top.lat, top.lon), 1),
        }
    return payload


@app.post("/api/countermeasures/test")
def test_countermeasure(req: CountermeasureReq) -> dict[str, Any]:
    if STATE.samples is None:
        raise HTTPException(status_code=400, detail="Load audio or a demo scenario first.")
    res = evaluate_countermeasure(
        STATE.samples,
        STATE.sr,
        t_cpa_s=req.t_cpa_s,
        f0_hz=req.f0_hz,
        speed_ms=req.speed_ms,
        d_cpa_m=req.d_cpa_m,
        method=req.method,
        param=req.param,
    )
    proc_spec = res.pop("spectrogram")
    res["spectrogram"] = _serialize_spectrogram(
        proc_spec,
        t_range=(max(0.0, req.t_cpa_s - 40.0), min(proc_spec.duration_s, req.t_cpa_s + 40.0)),
        f_range=(40.0, 360.0),
        max_w=460,
        max_h=200,
    )
    return res
