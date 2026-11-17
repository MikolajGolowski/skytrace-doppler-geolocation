"""Multi-stage acoustic Doppler geolocation pipeline (200m -> 100m -> 25m)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .adsb import FlightTrack
from .aircraft_types import should_keep_aircraft
from .dem import DEMProvider, FlatDEM
from .physics import (
    DEFAULT_SPEED_OF_SOUND,
    compute_track_cpa,
    enu_to_geodetic,
    geodetic_to_ecef,
    geodetic_to_enu,
    haversine_m,
    model_doppler_naive,
    model_doppler_retarded,
)
from .spectrogram import SpectrogramData


@dataclass
class EventQuery:
    """Measured acoustic & uncertainty bounds for a single aircraft flyover."""

    t_cpa_s: float
    f0_hz: float
    speed_min_ms: float
    speed_max_ms: float
    d_min_m: float
    d_max_m: float
    t_tolerance_s: float = 5.0
    f0_tolerance_hz: float = 4.0
    window_half_s: float = 40.0
    traced_points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class CandidateResult:
    """Scored candidate observer coordinate."""

    lat: float
    lon: float
    elevation_m: float
    flight1_id: str
    flight1_callsign: str
    flight1_type: str
    flight2_id: str | None
    flight2_callsign: str | None
    flight2_type: str | None
    score1: float
    score2: float
    total_score: float
    t_cpa1_utc: float
    t_cpa1_delayed_utc: float
    d_cpa1_m: float
    best_f0_1: float
    best_dt1_s: float
    t_cpa2_utc: float | None = None
    t_cpa2_delayed_utc: float | None = None
    d_cpa2_m: float | None = None
    best_f0_2: float | None = None
    rmse_hz: float | None = None

    def to_dict(self) -> dict:
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "elevation_m": round(self.elevation_m, 1),
            "flight1_id": self.flight1_id,
            "flight1_callsign": self.flight1_callsign,
            "flight1_type": self.flight1_type,
            "flight2_id": self.flight2_id,
            "flight2_callsign": self.flight2_callsign,
            "flight2_type": self.flight2_type,
            "score1": round(self.score1, 3),
            "score2": round(self.score2, 3),
            "total_score": round(self.total_score, 3),
            "t_cpa1_utc": round(self.t_cpa1_utc, 2),
            "t_cpa1_delayed_utc": round(self.t_cpa1_delayed_utc, 2),
            "d_cpa1_m": round(self.d_cpa1_m, 1),
            "best_f0_1": round(self.best_f0_1, 2),
            "best_dt1_s": round(self.best_dt1_s, 2),
            "t_cpa2_utc": round(self.t_cpa2_utc, 2) if self.t_cpa2_utc is not None else None,
            "t_cpa2_delayed_utc": round(self.t_cpa2_delayed_utc, 2) if self.t_cpa2_delayed_utc is not None else None,
            "d_cpa2_m": round(self.d_cpa2_m, 1) if self.d_cpa2_m is not None else None,
            "best_f0_2": round(self.best_f0_2, 2) if self.best_f0_2 is not None else None,
            "rmse_hz": round(self.rmse_hz, 3) if self.rmse_hz is not None else None,
        }


@dataclass
class GeolocationReport:
    """Full output of the geolocation pipeline."""

    mode: Literal["two_event", "single_event"]
    funnel: dict[str, int]
    top_candidates: list[CandidateResult]
    pair_peaks: list[CandidateResult]
    all_scored_sample: list[dict]
    fine_heatmap: list[dict]
    curve_overlays: list[dict]
    flight_paths: list[dict]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "funnel": self.funnel,
            "top_candidates": [c.to_dict() for c in self.top_candidates],
            "pair_peaks": [c.to_dict() for c in self.pair_peaks],
            "all_scored_sample": self.all_scored_sample,
            "fine_heatmap": self.fine_heatmap,
            "curve_overlays": self.curve_overlays,
            "flight_paths": self.flight_paths,
        }


def _track_matches_speed(track: FlightTrack, vmin: float, vmax: float) -> bool:
    if len(track.speeds_ms) == 0:
        return False
    return bool(np.any((track.speeds_ms >= vmin) & (track.speeds_ms <= vmax)))


def _build_corridor_grid_indices(
    tracks: list[FlightTrack],
    lat0: float,
    lon0: float,
    step_m: float,
    max_radius_m: float,
) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]]:
    """Build snapped integer lattice (ix, iy) points within `max_radius_m` of each track."""
    track_cells: dict[int, set[tuple[int, int]]] = {}
    all_cells: set[tuple[int, int]] = set()

    for idx, tr in enumerate(tracks):
        enu = geodetic_to_enu(tr.lats, tr.lons, np.zeros_like(tr.lats), lat0, lon0, 0.0)
        e_pts = enu[:, 0]
        n_pts = enu[:, 1]
        e_min = float(np.min(e_pts) - max_radius_m)
        e_max = float(np.max(e_pts) + max_radius_m)
        n_min = float(np.min(n_pts) - max_radius_m)
        n_max = float(np.max(n_pts) + max_radius_m)

        ix_range = np.arange(int(np.floor(e_min / step_m)), int(np.ceil(e_max / step_m)) + 1)
        iy_range = np.arange(int(np.floor(n_min / step_m)), int(np.ceil(n_max / step_m)) + 1)
        gxx, gyy = np.meshgrid(ix_range, iy_range, indexing="xy")
        gx = gxx.ravel() * step_m
        gy = gyy.ravel() * step_m

        # Subsample track points slightly if dense for quick 2D corridor distance check
        stride = max(1, len(e_pts) // 80)
        te = e_pts[::stride][None, :]
        tn = n_pts[::stride][None, :]
        d2 = (gx[:, None] - te) ** 2 + (gy[:, None] - tn) ** 2
        min_d = np.sqrt(np.min(d2, axis=1))
        keep = min_d <= max_radius_m

        cells = set(zip(gxx.ravel()[keep].tolist(), gyy.ravel()[keep].tolist()))
        track_cells[idx] = cells
        all_cells.update(cells)

    if not all_cells:
        return np.zeros((0, 2), dtype=np.int64), track_cells
    arr = np.array(sorted(all_cells), dtype=np.int64)
    return arr, track_cells


def _score_single_track_on_spec(
    spec: SpectrogramData,
    track: FlightTrack,
    obs_ecef: np.ndarray,
    event: EventQuery,
    t_cpa_del: float,
    dt_offset_s: float,
    c_sound: float,
) -> tuple[float, float]:
    """Score one track at fixed dt_offset_s by vectorized sweep of f0 within tolerance."""
    idx_mask = np.where(
        (spec.times >= event.t_cpa_s - event.window_half_s)
        & (spec.times <= event.t_cpa_s + event.window_half_s)
    )[0]
    if len(idx_mask) < 4:
        return 0.0, event.f0_hz

    t_audio = spec.times[idx_mask]
    t_utc_eval = t_audio - (event.t_cpa_s + dt_offset_s) + t_cpa_del
    unit_freqs, valid = model_doppler_retarded(
        track.timestamps,
        track.pos_ecef,
        track.vel_ecef,
        obs_ecef,
        t_utc_eval,
        f0=1.0,
        c=c_sound,
    )
    if np.sum(valid) < 4:
        return 0.0, event.f0_hz

    f0_grid = np.linspace(
        event.f0_hz - event.f0_tolerance_hz,
        event.f0_hz + event.f0_tolerance_hz,
        9,
    )
    t_idx_v = idx_mask[valid]
    u_v = unit_freqs[valid]

    mat = spec.smoothed
    df = float(spec.freqs[1] - spec.freqs[0])
    f_min = float(spec.freqs[0])
    n_freqs = len(spec.freqs)

    # Fundamental + 2nd harmonic vectorized across all f0 candidates
    f1 = f0_grid[:, None] * u_v[None, :]
    fy1 = np.rint((f1 - f_min) / df).astype(np.int64)
    in1 = (fy1 >= 0) & (fy1 < n_freqs)
    v1 = np.where(in1, mat[np.clip(fy1, 0, n_freqs - 1), t_idx_v[None, :]], 0.0)
    s1 = np.sum(v1, axis=1) / np.maximum(np.sum(in1, axis=1), 1)

    f2 = 2.0 * f1
    fy2 = np.rint((f2 - f_min) / df).astype(np.int64)
    in2 = (fy2 >= 0) & (fy2 < n_freqs)
    v2 = np.where(in2, mat[np.clip(fy2, 0, n_freqs - 1), t_idx_v[None, :]], 0.0)
    has2 = np.sum(in2, axis=1) > 0
    s2 = np.where(has2, np.sum(v2, axis=1) / np.maximum(np.sum(in2, axis=1), 1), 0.0)

    total = s1 + 0.25 * s2
    best_i = int(np.argmax(total))
    return float(total[best_i]), float(f0_grid[best_i])


def _score_two_events_locked(
    spec: SpectrogramData,
    track1: FlightTrack,
    track2: FlightTrack,
    obs_ecef: np.ndarray,
    event1: EventQuery,
    event2: EventQuery,
    t_del1: float,
    t_del2: float,
    c_sound: float,
) -> tuple[float, float, float, float, float, float]:
    """Colsto's locked two-event scoring: sweep dt1 & f0_1 on event1, lock event2's time."""
    dt_obs = event2.t_cpa_s - event1.t_cpa_s
    dt_phys = t_del2 - t_del1

    # Sweeping measurement error on Event 1's CPA time automatically positions Event 2!
    dt1_grid = np.linspace(-event1.t_tolerance_s, event1.t_tolerance_s, 11)
    best_total = -1e9
    best_s1 = 0.0
    best_s2 = 0.0
    best_f0_1 = event1.f0_hz
    best_f0_2 = event2.f0_hz
    best_dt1 = 0.0

    # Pre-check: locked offset error between physical delta_t and measured delta_t
    # dt2_effective = (event1.t_cpa_s + dt1 + dt_phys) - event2.t_cpa_s = dt1 + (dt_phys - dt_obs)
    delta_mismatch = dt_phys - dt_obs

    for dt1 in dt1_grid:
        dt2_locked = float(dt1) + delta_mismatch
        s1, f01 = _score_single_track_on_spec(
            spec, track1, obs_ecef, event1, t_del1, float(dt1), c_sound
        )
        s2, f02 = _score_single_track_on_spec(
            spec, track2, obs_ecef, event2, t_del2, dt2_locked, c_sound
        )
        tot = s1 + s2
        if tot > best_total:
            best_total = tot
            best_s1 = s1
            best_s2 = s2
            best_f0_1 = f01
            best_f0_2 = f02
            best_dt1 = float(dt1)

    return best_s1, best_s2, best_total, best_f0_1, best_f0_2, best_dt1


def _score_traced_single_event(
    spec: SpectrogramData,
    track: FlightTrack,
    obs_ecef: np.ndarray,
    event: EventQuery,
    t_del: float,
    c_sound: float,
) -> tuple[float, float, float, float]:
    """Score single event using traced curve RMSE (or spectrogram brightness if untraced)."""
    dt_grid = np.linspace(-event.t_tolerance_s, event.t_tolerance_s, 15)
    f0_grid = np.linspace(
        event.f0_hz - event.f0_tolerance_hz,
        event.f0_hz + event.f0_tolerance_hz,
        13,
    )

    if event.traced_points and len(event.traced_points) >= 3:
        t_pts = np.array([p[0] for p in event.traced_points], dtype=np.float64)
        f_pts = np.array([p[1] for p in event.traced_points], dtype=np.float64)
        best_score = -1e9
        best_rmse = 999.0
        best_f0 = event.f0_hz
        best_dt = 0.0

        for dt1 in dt_grid:
            t_utc = t_pts - (event.t_cpa_s + float(dt1)) + t_del
            unit_f, valid = model_doppler_retarded(
                track.timestamps,
                track.pos_ecef,
                track.vel_ecef,
                obs_ecef,
                t_utc,
                f0=1.0,
                c=c_sound,
            )
            if np.sum(valid) < max(3, int(len(t_pts) * 0.6)):
                continue
            u_v = unit_f[valid]
            f_v = f_pts[valid]
            # Analytical optimal f0 for minimum MSE
            f0_opt = float(np.dot(u_v, f_v) / np.maximum(np.dot(u_v, u_v), 1e-9))
            f0_clamped = float(np.clip(f0_opt, f0_grid[0], f0_grid[-1]))
            rmse = float(np.sqrt(np.mean((u_v * f0_clamped - f_v) ** 2)))
            score = 100.0 / (1.0 + rmse)
            if score > best_score:
                best_score = score
                best_rmse = rmse
                best_f0 = f0_clamped
                best_dt = float(dt1)
        return best_score, best_f0, best_dt, best_rmse

    best_sc = -1e9
    best_f0 = event.f0_hz
    best_dt = 0.0
    for dt1 in dt_grid:
        sc, f01 = _score_single_track_on_spec(
            spec, track, obs_ecef, event, t_del, float(dt1), c_sound
        )
        if sc > best_sc:
            best_sc = sc
            best_f0 = f01
            best_dt = float(dt1)
    return best_sc, best_f0, best_dt, 0.0


def _build_curve_overlay(
    spec: SpectrogramData,
    cand: CandidateResult,
    track_map: dict[str, FlightTrack],
    event1: EventQuery,
    event2: EventQuery | None,
    c_sound: float,
) -> dict:
    """Compute retarded vs naive Doppler overlays for a top candidate."""
    obs_ecef = geodetic_to_ecef(cand.lat, cand.lon, cand.elevation_m)
    tr1 = track_map[cand.flight1_id]

    t_mask1 = (spec.times >= event1.t_cpa_s - event1.window_half_s) & (
        spec.times <= event1.t_cpa_s + event1.window_half_s
    )
    t_aud1 = spec.times[t_mask1]
    t_utc1 = t_aud1 - (event1.t_cpa_s + cand.best_dt1_s) + cand.t_cpa1_delayed_utc
    f_ret1, v1 = model_doppler_retarded(
        tr1.timestamps, tr1.pos_ecef, tr1.vel_ecef, obs_ecef, t_utc1, cand.best_f0_1, c=c_sound
    )
    f_nai1, vn1 = model_doppler_naive(
        tr1.timestamps, tr1.pos_ecef, tr1.vel_ecef, obs_ecef, t_utc1, cand.best_f0_1, c=c_sound
    )

    overlay: dict = {
        "candidate": cand.to_dict(),
        "event1": {
            "t_audio": [round(float(x), 2) for x in t_aud1[v1]],
            "f_retarded": [round(float(x), 2) for x in f_ret1[v1]],
            "f_naive": [round(float(x), 2) if np.isfinite(x) else None for x in f_nai1[v1]],
        },
        "event2": None,
    }

    if event2 is not None and cand.flight2_id and cand.t_cpa2_delayed_utc is not None and cand.best_f0_2 is not None:
        tr2 = track_map[cand.flight2_id]
        dt_phys = cand.t_cpa2_delayed_utc - cand.t_cpa1_delayed_utc
        t_audio2_locked = event1.t_cpa_s + cand.best_dt1_s + dt_phys
        t_mask2 = (spec.times >= event2.t_cpa_s - event2.window_half_s) & (
            spec.times <= event2.t_cpa_s + event2.window_half_s
        )
        t_aud2 = spec.times[t_mask2]
        t_utc2 = t_aud2 - t_audio2_locked + cand.t_cpa2_delayed_utc
        f_ret2, v2 = model_doppler_retarded(
            tr2.timestamps, tr2.pos_ecef, tr2.vel_ecef, obs_ecef, t_utc2, cand.best_f0_2, c=c_sound
        )
        f_nai2, _ = model_doppler_naive(
            tr2.timestamps, tr2.pos_ecef, tr2.vel_ecef, obs_ecef, t_utc2, cand.best_f0_2, c=c_sound
        )
        overlay["event2"] = {
            "t_audio": [round(float(x), 2) for x in t_aud2[v2]],
            "f_retarded": [round(float(x), 2) for x in f_ret2[v2]],
            "f_naive": [round(float(x), 2) if np.isfinite(x) else None for x in f_nai2[v2]],
        }
    return overlay


def run_geolocation_pipeline(
    tracks: list[FlightTrack],
    spec: SpectrogramData,
    event1: EventQuery,
    event2: EventQuery | None = None,
    mode: Literal["two_event", "single_event"] = "two_event",
    dem: DEMProvider | None = None,
    aircraft_filter_mode: Literal["exclude_jets", "prop_only", "all"] = "exclude_jets",
    known_utc_cpa1: float | None = None,
    known_utc_tolerance_s: float = 35.0,
    grid_step_m: float = 200.0,
    refine_step_m: float = 100.0,
    fine_heatmap_step_m: float = 25.0,
    audible_radius_m: float = 10000.0,
    silence_margin_s: float = 35.0,
    enable_silence_filter: bool = True,
    c_sound: float = DEFAULT_SPEED_OF_SOUND,
    progress_cb: Callable[[str, float], None] | None = None,
) -> GeolocationReport:
    """Execute the full Colsto-style acoustic Doppler geolocation pipeline."""
    if dem is None:
        dem = FlatDEM(0.0)
    if mode == "two_event" and event2 is None:
        raise ValueError("two_event mode requires both event1 and event2.")

    def report_prog(msg: str, frac: float) -> None:
        if progress_cb:
            progress_cb(msg, frac)

    report_prog("Filtering aircraft by ICAO type code", 0.05)
    total_tracks = len(tracks)
    non_jet_tracks = [t for t in tracks if should_keep_aircraft(t.type_code, aircraft_filter_mode)]
    track_map = {t.icao24: t for t in tracks}

    # Reference origin for local ENU lattice
    if non_jet_tracks:
        lat0 = float(np.mean([np.mean(t.lats) for t in non_jet_tracks]))
        lon0 = float(np.mean([np.mean(t.lons) for t in non_jet_tracks]))
    elif tracks:
        lat0 = float(np.mean([np.mean(t.lats) for t in tracks]))
        lon0 = float(np.mean([np.mean(t.lons) for t in tracks]))
    else:
        return GeolocationReport(
            mode=mode,
            funnel={"total_tracks": 0},
            top_candidates=[],
            pair_peaks=[],
            all_scored_sample=[],
            fine_heatmap=[],
            curve_overlays=[],
            flight_paths=[],
        )

    # Speed filtering
    cand1_tracks = [
        t for t in non_jet_tracks if _track_matches_speed(t, event1.speed_min_ms, event1.speed_max_ms)
    ]
    if mode == "single_event" and known_utc_cpa1 is not None:
        cand1_tracks = [
            t
            for t in cand1_tracks
            if t.start_time - known_utc_tolerance_s - 40.0 <= known_utc_cpa1 <= t.end_time + known_utc_tolerance_s + 40.0
        ]

    cand2_tracks: list[FlightTrack] = []
    if mode == "two_event" and event2 is not None:
        cand2_tracks = [
            t for t in non_jet_tracks if _track_matches_speed(t, event2.speed_min_ms, event2.speed_max_ms)
        ]

    report_prog("Building 200m candidate spatial grid", 0.18)
    grid1_cells, t1_cell_map = _build_corridor_grid_indices(
        cand1_tracks, lat0, lon0, grid_step_m, event1.d_max_m
    )

    if len(grid1_cells) == 0:
        return GeolocationReport(
            mode=mode,
            funnel={
                "total_tracks": total_tracks,
                "non_jet_tracks": len(non_jet_tracks),
                "event1_speed_tracks": len(cand1_tracks),
                "event2_speed_tracks": len(cand2_tracks),
                "grid_200m_event1": 0,
                "after_time_offset": 0,
                "after_silence_filter": 0,
                "refined_100m_scored": 0,
            },
            top_candidates=[],
            pair_peaks=[],
            all_scored_sample=[],
            fine_heatmap=[],
            curve_overlays=[],
            flight_paths=[],
        )

    # Convert Event 1 corridor cells to lat/lon/elev
    enu1 = np.column_stack([grid1_cells * grid_step_m, np.zeros(len(grid1_cells))])
    lats1, lons1, _ = enu_to_geodetic(enu1, lat0, lon0, 0.0)
    elevs1 = dem.get_elevations(lats1, lons1)
    ecefs1 = geodetic_to_ecef(lats1, lons1, elevs1)

    # Map cell -> index in grid1_cells
    cell_to_idx = {(int(c[0]), int(c[1])): i for i, c in enumerate(grid1_cells)}

    # Check exact 3D slant distance for Event 1 grid points
    report_prog("Computing 3D slant distance & delayed CPA times", 0.30)
    event1_hits: dict[int, list[tuple[int, float, float, float]]] = {}  # cell_idx -> [(t1_idx, t_cpa, d_cpa, t_del)]
    for t_idx, tr in enumerate(cand1_tracks):
        cells = t1_cell_map[t_idx]
        for cell in cells:
            ci = cell_to_idx[cell]
            t_cpa, d_cpa, t_del = compute_track_cpa(
                tr.timestamps, tr.pos_ecef, tr.vel_ecef, ecefs1[ci], c=c_sound
            )
            if event1.d_min_m <= d_cpa <= event1.d_max_m:
                if mode == "single_event" and known_utc_cpa1 is not None:
                    if abs(t_del - known_utc_cpa1) > known_utc_tolerance_s:
                        continue
                event1_hits.setdefault(ci, []).append((t_idx, t_cpa, d_cpa, t_del))

    n_grid_event1 = len(event1_hits)

    # Stage 4: Time offset & Event 2 slant distance filter (or single-event pass-through)
    surviving_pairs: list[tuple[int, tuple[int, float, float, float], tuple[int, float, float, float] | None]] = []

    if mode == "two_event" and event2 is not None:
        report_prog("Applying inter-event time offset & Event 2 slant filter", 0.45)
        dt_obs = event2.t_cpa_s - event1.t_cpa_s
        dt_tol = event1.t_tolerance_s + event2.t_tolerance_s + 4.0
        _, t2_cell_map = _build_corridor_grid_indices(
            cand2_tracks, lat0, lon0, grid_step_m, event2.d_max_m
        )
        # Invert cell -> event2 tracks that cover it
        cell_to_t2: dict[tuple[int, int], list[int]] = {}
        for t2_idx, cells in t2_cell_map.items():
            for cell in cells:
                cell_to_t2.setdefault(cell, []).append(t2_idx)

        for ci, hits1 in event1_hits.items():
            cell = (int(grid1_cells[ci, 0]), int(grid1_cells[ci, 1]))
            t2_list = cell_to_t2.get(cell, [])
            if not t2_list:
                continue
            obs_ecef = ecefs1[ci]
            for t2_idx in t2_list:
                tr2 = cand2_tracks[t2_idx]
                t_cpa2, d_cpa2, t_del2 = compute_track_cpa(
                    tr2.timestamps, tr2.pos_ecef, tr2.vel_ecef, obs_ecef, c=c_sound
                )
                if not (event2.d_min_m <= d_cpa2 <= event2.d_max_m):
                    continue
                for h1 in hits1:
                    _, _, _, t_del1 = h1
                    if cand1_tracks[h1[0]].icao24 == tr2.icao24 and abs(t_del2 - t_del1) < 30.0:
                        continue
                    if abs((t_del2 - t_del1) - dt_obs) <= dt_tol:
                        surviving_pairs.append((ci, h1, (t2_idx, t_cpa2, d_cpa2, t_del2)))
    else:
        for ci, hits1 in event1_hits.items():
            for h1 in hits1:
                surviving_pairs.append((ci, h1, None))

    n_after_offset = len({p[0] for p in surviving_pairs})

    # Stage 5: Silence constraint filter (between Event 1 and Event 2)
    if mode == "two_event" and enable_silence_filter and surviving_pairs:
        report_prog("Applying inter-event silence constraint filter", 0.58)
        # Precompute track bounding boxes in ENU for fast audible-radius culling
        # (Check non-jet tracks or low-altitude aircraft < 6500m MSL)
        silence_tracks = [
            tr for tr in tracks
            if (tr in non_jet_tracks) or float(np.min(tr.altitudes_m)) < 6500.0
        ]
        track_bounds = []
        for tr in silence_tracks:
            enu_t = geodetic_to_enu(tr.lats, tr.lons, np.zeros_like(tr.lats), lat0, lon0, 0.0)
            track_bounds.append(
                (
                    tr,
                    float(np.min(enu_t[:, 0]) - audible_radius_m),
                    float(np.max(enu_t[:, 0]) + audible_radius_m),
                    float(np.min(enu_t[:, 1]) - audible_radius_m),
                    float(np.max(enu_t[:, 1]) + audible_radius_m),
                )
            )

        filtered_pairs = []
        silence_cache: dict[tuple[int, float, float], bool] = {}

        for ci, h1, h2 in surviving_pairs:
            assert h2 is not None
            t_del1 = h1[3]
            t_del2 = h2[3]
            eff_margin = min(silence_margin_s, abs(t_del2 - t_del1) * 0.22)
            w_start = min(t_del1, t_del2) + eff_margin
            w_end = max(t_del1, t_del2) - eff_margin
            if w_end <= w_start:
                filtered_pairs.append((ci, h1, h2))
                continue

            cache_key = (ci, round(w_start, 0), round(w_end, 0))
            if cache_key not in silence_cache:
                ex = float(grid1_cells[ci, 0] * grid_step_m)
                ny = float(grid1_cells[ci, 1] * grid_step_m)
                obs_ecef = ecefs1[ci]
                f1_id = cand1_tracks[h1[0]].icao24
                f2_id = cand2_tracks[h2[0]].icao24
                heard_intruder = False
                for tr, xmin, xmax, ymin, ymax in track_bounds:
                    if tr.icao24 == f1_id or tr.icao24 == f2_id:
                        continue
                    if not (xmin <= ex <= xmax and ymin <= ny <= ymax):
                        continue
                    if tr.end_time < w_start - 30.0 or tr.start_time > w_end + 30.0:
                        continue
                    _, d_int, t_del_int = compute_track_cpa(
                        tr.timestamps, tr.pos_ecef, tr.vel_ecef, obs_ecef, c=c_sound
                    )
                    if d_int <= audible_radius_m and (w_start <= t_del_int <= w_end):
                        heard_intruder = True
                        break
                silence_cache[cache_key] = not heard_intruder

            if silence_cache[cache_key]:
                filtered_pairs.append((ci, h1, h2))
        surviving_pairs = filtered_pairs

    n_after_silence = len({p[0] for p in surviving_pairs})

    # Stage 6 & 7: 200m -> 100m Grid Refinement & Retarded-Time Locked Scoring
    report_prog("Refining to 100m grid & scoring retarded-time Doppler curves", 0.72)

    # Sub-offsets to expand each 200m cell into 100m sub-cells (-0.25, +0.25 of grid_step_m)
    sub_ratio = max(1, int(round(grid_step_m / refine_step_m)))
    if sub_ratio >= 2:
        offsets_1d = (np.arange(sub_ratio) - (sub_ratio - 1) / 2.0) * refine_step_m
        ox, oy = np.meshgrid(offsets_1d, offsets_1d, indexing="xy")
        sub_offsets = np.column_stack([ox.ravel(), oy.ravel()])
    else:
        sub_offsets = np.array([[0.0, 0.0]])

    # Cap extreme explosions gracefully while keeping closest time-offset cells
    max_coarse_pairs = 650
    if len(surviving_pairs) > max_coarse_pairs:
        if mode == "two_event" and event2 is not None:
            dt_obs = event2.t_cpa_s - event1.t_cpa_s
            surviving_pairs.sort(key=lambda p: abs((p[2][3] - p[1][3]) - dt_obs))  # type: ignore[index]
        elif mode == "single_event" and known_utc_cpa1 is not None:
            surviving_pairs.sort(key=lambda p: abs(p[1][3] - known_utc_cpa1))
        surviving_pairs = surviving_pairs[:max_coarse_pairs]

    # Collect all 100m sub-grid coordinates in one batch for DEM lookup
    sub_enu_list: list[np.ndarray] = []
    sub_pair_meta: list[tuple[int, int, int | None]] = []
    for ci, h1, h2 in surviving_pairs:
        cx = float(grid1_cells[ci, 0] * grid_step_m)
        cy = float(grid1_cells[ci, 1] * grid_step_m)
        for dx, dy in sub_offsets:
            sub_enu_list.append([cx + dx, cy + dy, 0.0])
            sub_pair_meta.append((h1[0], h2[0] if h2 is not None else None, ci))

    scored_results: list[CandidateResult] = []
    if sub_enu_list:
        sub_enu = np.array(sub_enu_list, dtype=np.float64)
        s_lats, s_lons, _ = enu_to_geodetic(sub_enu, lat0, lon0, 0.0)
        s_elevs = dem.get_elevations(s_lats, s_lons)
        s_ecefs = geodetic_to_ecef(s_lats, s_lons, s_elevs)

        for k, (t1_idx, t2_idx, _) in enumerate(sub_pair_meta):
            tr1 = cand1_tracks[t1_idx]
            obs_ecef = s_ecefs[k]
            lat_k = float(s_lats[k])
            lon_k = float(s_lons[k])
            elev_k = float(s_elevs[k])
            t_cpa1, d_cpa1, t_del1 = compute_track_cpa(
                tr1.timestamps, tr1.pos_ecef, tr1.vel_ecef, obs_ecef, c=c_sound
            )

            if mode == "two_event" and event2 is not None and t2_idx is not None:
                tr2 = cand2_tracks[t2_idx]
                t_cpa2, d_cpa2, t_del2 = compute_track_cpa(
                    tr2.timestamps, tr2.pos_ecef, tr2.vel_ecef, obs_ecef, c=c_sound
                )
                s1, s2, tot, f01, f02, dt1 = _score_two_events_locked(
                    spec, tr1, tr2, obs_ecef, event1, event2, t_del1, t_del2, c_sound
                )
                scored_results.append(
                    CandidateResult(
                        lat=lat_k,
                        lon=lon_k,
                        elevation_m=elev_k,
                        flight1_id=tr1.icao24,
                        flight1_callsign=tr1.callsign or tr1.icao24,
                        flight1_type=tr1.type_code or "UNK",
                        flight2_id=tr2.icao24,
                        flight2_callsign=tr2.callsign or tr2.icao24,
                        flight2_type=tr2.type_code or "UNK",
                        score1=s1,
                        score2=s2,
                        total_score=tot,
                        t_cpa1_utc=t_cpa1,
                        t_cpa1_delayed_utc=t_del1,
                        d_cpa1_m=d_cpa1,
                        best_f0_1=f01,
                        best_dt1_s=dt1,
                        t_cpa2_utc=t_cpa2,
                        t_cpa2_delayed_utc=t_del2,
                        d_cpa2_m=d_cpa2,
                        best_f0_2=f02,
                    )
                )
            else:
                sc, f01, dt1, rmse = _score_traced_single_event(
                    spec, tr1, obs_ecef, event1, t_del1, c_sound
                )
                scored_results.append(
                    CandidateResult(
                        lat=lat_k,
                        lon=lon_k,
                        elevation_m=elev_k,
                        flight1_id=tr1.icao24,
                        flight1_callsign=tr1.callsign or tr1.icao24,
                        flight1_type=tr1.type_code or "UNK",
                        flight2_id=None,
                        flight2_callsign=None,
                        flight2_type=None,
                        score1=sc,
                        score2=0.0,
                        total_score=sc,
                        t_cpa1_utc=t_cpa1,
                        t_cpa1_delayed_utc=t_del1,
                        d_cpa1_m=d_cpa1,
                        best_f0_1=f01,
                        best_dt1_s=dt1,
                        rmse_hz=rmse if event1.traced_points else None,
                    )
                )

    scored_results.sort(key=lambda r: r.total_score, reverse=True)

    # Group by flight pair -> take peak candidate per pair (Colsto's scatter plot clustering)
    pair_best: dict[tuple[str, str | None], CandidateResult] = {}
    for res in scored_results:
        key = (res.flight1_id, res.flight2_id)
        if key not in pair_best or res.total_score > pair_best[key].total_score:
            pair_best[key] = res
    pair_peaks = sorted(pair_best.values(), key=lambda r: r.total_score, reverse=True)

    # Fine 25m resolution heatmap around #1 candidate for winning flight pair
    report_prog("Generating 25m fine resolution heatmap around top candidate", 0.90)
    fine_heatmap: list[dict] = []
    if pair_peaks:
        winner = pair_peaks[0]
        tr1_win = track_map[winner.flight1_id]
        tr2_win = track_map[winner.flight2_id] if winner.flight2_id else None
        half_span = 275.0
        steps = np.arange(-half_span, half_span + fine_heatmap_step_m * 0.5, fine_heatmap_step_m)
        hx, hy = np.meshgrid(steps, steps, indexing="xy")
        fine_enu = np.column_stack([hx.ravel(), hy.ravel(), np.zeros(hx.size)])
        f_lats, f_lons, _ = enu_to_geodetic(fine_enu, winner.lat, winner.lon, 0.0)
        f_elevs = dem.get_elevations(f_lats, f_lons)
        f_ecefs = geodetic_to_ecef(f_lats, f_lons, f_elevs)

        best_fine_cand = winner
        for idx_f in range(len(f_lats)):
            obs_f = f_ecefs[idx_f]
            tc1, dc1, td1 = compute_track_cpa(
                tr1_win.timestamps, tr1_win.pos_ecef, tr1_win.vel_ecef, obs_f, c=c_sound
            )
            if mode == "two_event" and tr2_win is not None and event2 is not None:
                tc2, dc2, td2 = compute_track_cpa(
                    tr2_win.timestamps, tr2_win.pos_ecef, tr2_win.vel_ecef, obs_f, c=c_sound
                )
                s1, s2, tot, f01, f02, dt1 = _score_two_events_locked(
                    spec, tr1_win, tr2_win, obs_f, event1, event2, td1, td2, c_sound
                )
                fine_heatmap.append(
                    {
                        "lat": round(float(f_lats[idx_f]), 6),
                        "lon": round(float(f_lons[idx_f]), 6),
                        "score": round(float(tot), 3),
                        "score1": round(float(s1), 3),
                        "score2": round(float(s2), 3),
                    }
                )
                if tot > best_fine_cand.total_score:
                    best_fine_cand = CandidateResult(
                        lat=float(f_lats[idx_f]),
                        lon=float(f_lons[idx_f]),
                        elevation_m=float(f_elevs[idx_f]),
                        flight1_id=tr1_win.icao24,
                        flight1_callsign=tr1_win.callsign or tr1_win.icao24,
                        flight1_type=tr1_win.type_code or "UNK",
                        flight2_id=tr2_win.icao24,
                        flight2_callsign=tr2_win.callsign or tr2_win.icao24,
                        flight2_type=tr2_win.type_code or "UNK",
                        score1=s1,
                        score2=s2,
                        total_score=tot,
                        t_cpa1_utc=tc1,
                        t_cpa1_delayed_utc=td1,
                        d_cpa1_m=dc1,
                        best_f0_1=f01,
                        best_dt1_s=dt1,
                        t_cpa2_utc=tc2,
                        t_cpa2_delayed_utc=td2,
                        d_cpa2_m=dc2,
                        best_f0_2=f02,
                    )
            else:
                sc, f01, dt1, rmse = _score_traced_single_event(
                    spec, tr1_win, obs_f, event1, td1, c_sound
                )
                fine_heatmap.append(
                    {
                        "lat": round(float(f_lats[idx_f]), 6),
                        "lon": round(float(f_lons[idx_f]), 6),
                        "score": round(float(sc), 3),
                        "score1": round(float(sc), 3),
                        "score2": 0.0,
                    }
                )
                if sc > best_fine_cand.total_score:
                    best_fine_cand = CandidateResult(
                        lat=float(f_lats[idx_f]),
                        lon=float(f_lons[idx_f]),
                        elevation_m=float(f_elevs[idx_f]),
                        flight1_id=tr1_win.icao24,
                        flight1_callsign=tr1_win.callsign or tr1_win.icao24,
                        flight1_type=tr1_win.type_code or "UNK",
                        flight2_id=None,
                        flight2_callsign=None,
                        flight2_type=None,
                        score1=sc,
                        score2=0.0,
                        total_score=sc,
                        t_cpa1_utc=tc1,
                        t_cpa1_delayed_utc=td1,
                        d_cpa1_m=dc1,
                        best_f0_1=f01,
                        best_dt1_s=dt1,
                        rmse_hz=rmse if event1.traced_points else None,
                    )

        # Insert refined 25m peak at #1 if it improved over the 100m point
        if haversine_m(best_fine_cand.lat, best_fine_cand.lon, pair_peaks[0].lat, pair_peaks[0].lon) > 1.0:
            pair_peaks[0] = best_fine_cand
            scored_results.insert(0, best_fine_cand)

    top_candidates = pair_peaks[:15]
    curve_overlays = [
        _build_curve_overlay(spec, cand, track_map, event1, event2, c_sound)
        for cand in pair_peaks[:3]
    ]

    # Collect flight paths relevant to top candidates for map visualization
    featured_ids: list[str] = []
    for c in pair_peaks[:5]:
        if c.flight1_id not in featured_ids:
            featured_ids.append(c.flight1_id)
        if c.flight2_id and c.flight2_id not in featured_ids:
            featured_ids.append(c.flight2_id)
    for tr in non_jet_tracks[:20]:
        if tr.icao24 not in featured_ids:
            featured_ids.append(tr.icao24)

    flight_paths = []
    for fid in featured_ids:
        tr = track_map[fid]
        stride = max(1, len(tr.lats) // 60)
        flight_paths.append(
            {
                "icao24": tr.icao24,
                "callsign": tr.callsign or tr.icao24,
                "type_code": tr.type_code or "UNK",
                "propulsion_class": tr.propulsion_class,
                "coords": [
                    [round(float(la), 5), round(float(lo), 5)]
                    for la, lo in zip(tr.lats[::stride], tr.lons[::stride])
                ],
            }
        )

    funnel = {
        "total_tracks": total_tracks,
        "non_jet_tracks": len(non_jet_tracks),
        "event1_speed_tracks": len(cand1_tracks),
        "event2_speed_tracks": len(cand2_tracks),
        "grid_200m_event1": n_grid_event1,
        "after_time_offset": n_after_offset,
        "after_silence_filter": n_after_silence,
        "refined_100m_scored": len(scored_results),
    }
    report_prog("Completed geolocation pipeline", 1.0)

    return GeolocationReport(
        mode=mode,
        funnel=funnel,
        top_candidates=top_candidates,
        pair_peaks=pair_peaks,
        all_scored_sample=[r.to_dict() for r in scored_results[:400]],
        fine_heatmap=fine_heatmap,
        curve_overlays=curve_overlays,
        flight_paths=flight_paths,
    )
