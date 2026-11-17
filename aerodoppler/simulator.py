"""Realistic synthetic acoustic + ADS-B benchmark scenario generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adsb import FlightTrack
from .dem import DEMProvider, SyntheticTerrainDEM
from .physics import (
    DEFAULT_SPEED_OF_SOUND,
    enu_to_geodetic,
    geodetic_to_ecef,
    model_doppler_retarded,
)


@dataclass
class DemoScenario:
    """Generated synthetic benchmark scenario with hidden ground truth."""

    name: str
    mode: str
    samples: np.ndarray
    sample_rate: int
    tracks: list[FlightTrack]
    dem: DEMProvider
    true_lat: float
    true_lon: float
    true_elev_m: float
    recording_start_utc: float
    true_event1_audio_s: float
    true_event2_audio_s: float | None
    true_f0_1: float
    true_f0_2: float | None
    known_utc_cpa1: float | None


def _make_curved_track(
    icao24: str,
    callsign: str,
    type_code: str,
    lat0: float,
    lon0: float,
    t_start: float,
    duration_s: float,
    start_enu: tuple[float, float, float],
    heading_deg: float,
    speed_start: float,
    speed_end: float,
    turn_rate_deg_s: float = 0.0,
    climb_rate_ms: float = 0.0,
    dt_step: float = 1.0,
) -> FlightTrack:
    times = np.arange(t_start, t_start + duration_s + 0.5, dt_step, dtype=np.float64)
    n = len(times)
    e = np.zeros(n, dtype=np.float64)
    no = np.zeros(n, dtype=np.float64)
    u = np.zeros(n, dtype=np.float64)
    e[0], no[0], u[0] = start_enu

    hdg = np.deg2rad(heading_deg)
    turn_rad = np.deg2rad(turn_rate_deg_s)
    for i in range(1, n):
        frac = i / max(1, n - 1)
        spd = speed_start + (speed_end - speed_start) * frac
        hdg += turn_rad * dt_step
        e[i] = e[i - 1] + spd * np.sin(hdg) * dt_step
        no[i] = no[i - 1] + spd * np.cos(hdg) * dt_step
        u[i] = u[i - 1] + climb_rate_ms * dt_step

    lats, lons, alts = enu_to_geodetic(np.column_stack([e, no, u]), lat0, lon0, 0.0)
    return FlightTrack(
        icao24=icao24,
        callsign=callsign,
        type_code=type_code,
        timestamps=times,
        lats=lats,
        lons=lons,
        altitudes_m=alts,
    )


def _synthesize_flyover_signal(
    t_audio: np.ndarray,
    track: FlightTrack,
    obs_ecef: np.ndarray,
    recording_start_utc: float,
    f0_hz: float,
    amplitude: float,
    c_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> np.ndarray:
    """Synthesize physically exact retarded-time propeller harmonics for a 3D trajectory."""
    t_utc = recording_start_utc + t_audio
    freqs, valid = model_doppler_retarded(
        track.timestamps,
        track.pos_ecef,
        track.vel_ecef,
        obs_ecef,
        t_utc,
        f0=f0_hz,
        c=c_sound,
    )
    sig = np.zeros_like(t_audio)
    if not np.any(valid):
        return sig

    f_clean = np.where(valid, freqs, f0_hz)
    dt = float(t_audio[1] - t_audio[0])
    phase = 2.0 * np.pi * np.cumsum(f_clean) * dt

    # Compute retarded distance attenuation envelope (1 / r)
    diff = track.pos_ecef - obs_ecef
    dists = np.linalg.norm(diff, axis=1)
    t_heard_track = track.timestamps + dists / c_sound
    dist_interp = np.interp(t_utc, t_heard_track, dists, left=1e6, right=1e6)
    min_d = max(float(np.min(dist_interp[valid])), 150.0)
    env = np.where(valid, (min_d / np.maximum(dist_interp, 150.0)) ** 1.35, 0.0)
    env *= np.where(env > 0.05, 1.0, 0.0)

    # Fundamental + 2nd & 3rd propeller blade-pass harmonics
    wave = (
        1.0 * np.sin(phase)
        + 0.42 * np.sin(2.0 * phase + 0.4)
        + 0.18 * np.sin(3.0 * phase + 1.1)
    )
    return amplitude * env * wave


def _ambient_background(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Generate realistic pink-ish outdoor wind noise + speech-like formants + 60Hz hum."""
    white = rng.standard_normal(n_samples)
    # 1/f-ish smoothing via simple IIR filter
    b = np.array([0.0499, -0.0959, 0.0506, -0.0044])
    a = np.array([1.0, -2.4949, 2.0172, -0.5221])
    from scipy.signal import lfilter

    pink = lfilter(b, a, white)
    pink /= max(float(np.std(pink)), 1e-6)

    t = np.arange(n_samples, dtype=np.float64) / sr
    # Faint 60 Hz + 120 Hz stationary electrical hum (tests whitening!)
    hum = 0.08 * np.sin(2 * np.pi * 60.0 * t) + 0.04 * np.sin(2 * np.pi * 120.0 * t)

    # Occasional speech-like bursts (150-350 Hz modulated formants)
    speech = np.zeros(n_samples, dtype=np.float64)
    for burst_t in np.arange(8.0, t[-1] - 5.0, 17.0):
        mask = (t >= burst_t) & (t <= burst_t + 3.2)
        f_voice = 165.0 + 25.0 * np.sin(2 * np.pi * 1.4 * t[mask])
        speech[mask] += 0.22 * np.sin(2 * np.pi * f_voice * t[mask]) * np.hanning(np.sum(mask))

    return 0.24 * pink + hum + speech


def generate_demo_scenario(
    preset: str = "quick_two_event",
    seed: int = 42,
) -> DemoScenario:
    """Generate realistic end-to-end audio + ADS-B scenarios.

    Presets:
    - 'quick_two_event': 160s audio, 2 curved prop flyovers 90s apart + decoy airport with silence breaker.
    - 'colsto_920s': 1020s (17-min) audio, 2 prop flyovers 920s apart matching Colsto's video offset!
    - 'single_1min_faint': 60s audio, single nearly-inaudible prop flyover with known UTC time.
    """
    rng = np.random.default_rng(seed)
    sr = 2000
    t0_utc = 1725827800.0  # Sept 8 UTC epoch

    # True observer near Long Island Sound (paying tribute to Colsto's 40.87, -73.50)
    true_lat = 40.8724
    true_lon = -73.5038
    dem = SyntheticTerrainDEM(base_m=12.0, hill_amp_m=55.0, seed_lat=true_lat, seed_lon=true_lon)
    true_elev = float(dem.get_elevations(np.array([true_lat]), np.array([true_lon]))[0])
    obs_ecef = geodetic_to_ecef(true_lat, true_lon, true_elev)

    tracks: list[FlightTrack] = []

    if preset == "single_1min_faint":
        duration_s = 60.0
        n_samples = int(duration_s * sr)
        t_audio = np.arange(n_samples, dtype=np.float64) / sr

        # Single non-linear prop passing at ~t_audio = 30s
        target1 = _make_curved_track(
            icao24="a4f120",
            callsign="N718SR",
            type_code="SR22",
            lat0=true_lat,
            lon0=true_lon,
            t_start=t0_utc - 25.0,
            duration_s=115.0,
            start_enu=(-3900.0, -700.0, 920.0),
            heading_deg=72.0,
            speed_start=78.0,
            speed_end=102.0,
            turn_rate_deg_s=1.45,
            climb_rate_ms=2.2,
        )
        tracks.append(target1)

        # Decoy prop planes elsewhere in region at same timestamp & similar speed
        tracks.append(
            _make_curved_track(
                "a90011", "N204DC", "C182", true_lat, true_lon,
                t0_utc - 25.0, 115.0, (-14000.0, 9500.0, 1100.0), 115.0, 86.0, 88.0, turn_rate_deg_s=-0.1
            )
        )
        tracks.append(
            _make_curved_track(
                "a90022", "N511PA", "PA28", true_lat, true_lon,
                t0_utc - 25.0, 115.0, (11000.0, -8500.0, 1400.0), 35.0, 78.0, 82.0
            )
        )
        # Fast commercial jets passing nearby (filtered by ICAO jet exclusion)
        tracks.append(
            _make_curved_track(
                "a11b73", "DAL412", "B738", true_lat, true_lon,
                t0_utc - 30.0, 120.0, (-18000.0, 2000.0, 9800.0), 88.0, 225.0, 228.0
            )
        )

        audio = _ambient_background(n_samples, sr, rng)
        f0_1 = 136.5
        audio += _synthesize_flyover_signal(t_audio, target1, obs_ecef, t0_utc, f0_1, amplitude=0.055)
        audio = np.clip(audio, -0.98, 0.98)

        # Exact true delayed CPA in audio frame
        from .physics import compute_track_cpa
        _, _, t_del1 = compute_track_cpa(target1.timestamps, target1.pos_ecef, target1.vel_ecef, obs_ecef)
        return DemoScenario(
            name="Single-Event 1-Min Faint Clip",
            mode="single_event",
            samples=audio,
            sample_rate=sr,
            tracks=tracks,
            dem=dem,
            true_lat=true_lat,
            true_lon=true_lon,
            true_elev_m=true_elev,
            recording_start_utc=t0_utc,
            true_event1_audio_s=float(t_del1 - t0_utc),
            true_event2_audio_s=None,
            true_f0_1=f0_1,
            true_f0_2=None,
            known_utc_cpa1=float(t_del1),
        )

    # Two-event modes: quick_two_event (90s gap) or colsto_920s (920s gap)
    gap_s = 920.0 if preset == "colsto_920s" else 90.0
    t1_offset = 45.0
    t2_offset = t1_offset + gap_s
    duration_s = t2_offset + 45.0
    n_samples = int(duration_s * sr)
    t_audio = np.arange(n_samples, dtype=np.float64) / sr

    # Target Flight 1: curved turning & accelerating Cessna 182 around t = 45s
    target1 = _make_curved_track(
        icao24="a418ct",
        callsign="N418CT",
        type_code="C182",
        lat0=true_lat,
        lon0=true_lon,
        t_start=t0_utc + t1_offset - 50.0,
        duration_s=105.0,
        start_enu=(-3600.0, -450.0, 980.0),
        heading_deg=74.0,
        speed_start=72.0,
        speed_end=84.0,
        turn_rate_deg_s=0.22,
        climb_rate_ms=1.5,
    )

    # Target Flight 2: Piper Cherokee passing gap_s later on crossing heading
    target2 = _make_curved_track(
        icao24="a920pa",
        callsign="N920PA",
        type_code="PA28",
        lat0=true_lat,
        lon0=true_lon,
        t_start=t0_utc + t2_offset - 50.0,
        duration_s=105.0,
        start_enu=(900.0, -3200.0, 740.0),
        heading_deg=342.0,
        speed_start=66.0,
        speed_end=60.0,
        turn_rate_deg_s=-0.16,
        climb_rate_ms=-0.8,
    )
    tracks.extend([target1, target2])

    # Decoy Airport Cluster 16 km NE: two prop planes separated by EXACTLY gap_s,
    # PLUS an intruder prop right in the middle of the gap -> Silence filter eliminates this!
    airport_e, airport_n = 11500.0, 11500.0
    decoy1 = _make_curved_track(
        "ad0001", "N101AP", "C172", true_lat, true_lon,
        t0_utc + t1_offset - 50.0, 105.0,
        (airport_e - 3300.0, airport_n + 600.0, 820.0), 85.0, 76.0, 78.0
    )
    decoy2 = _make_curved_track(
        "ad0002", "N102AP", "BE36", true_lat, true_lon,
        t0_utc + t2_offset - 50.0, 105.0,
        (airport_e + 700.0, airport_n - 2900.0, 680.0), 350.0, 64.0, 63.0
    )
    decoy_silence_breaker = _make_curved_track(
        "ad0003", "N103BR", "PA28", true_lat, true_lon,
        t0_utc + (t1_offset + gap_s * 0.5) - 50.0, 105.0,
        (airport_e - 2800.0, airport_n - 400.0, 650.0), 75.0, 62.0, 65.0
    )
    tracks.extend([decoy1, decoy2, decoy_silence_breaker])

    # Another decoy prop pair 16 km SW with wrong curve geometry / speed or distance
    tracks.append(
        _make_curved_track(
            "ad0004", "N304SW", "SR22", true_lat, true_lon,
            t0_utc + t1_offset - 45.0, 95.0,
            (-13000.0, -11000.0, 1200.0), 295.0, 98.0, 102.0
        )
    )
    tracks.append(
        _make_curved_track(
            "ad0005", "N305SW", "C172", true_lat, true_lon,
            t0_utc + t2_offset - 45.0, 95.0,
            (-11000.0, -13500.0, 1100.0), 15.0, 58.0, 58.0
        )
    )

    # Commercial jets (excluded by ICAO type filter)
    for j_idx, (icao, call, tcode, start_e, start_n, hdg, alt) in enumerate(
        [
            ("ajet01", "JBU112", "A320", -16000.0, 1500.0, 80.0, 10200.0),
            ("ajet02", "UAL884", "B738", 12000.0, -14000.0, 320.0, 9500.0),
            ("ajet03", "AAL205", "B77W", -12000.0, -6000.0, 50.0, 11200.0),
            ("ajet04", "N900GF", "GLF5", 5000.0, 14000.0, 210.0, 12500.0),
        ]
    ):
        tracks.append(
            _make_curved_track(
                icao, call, tcode, true_lat, true_lon,
                t0_utc + t1_offset - 30.0 + j_idx * 15.0, 120.0,
                (start_e, start_n, alt), hdg, 225.0, 230.0
            )
        )

    audio = _ambient_background(n_samples, sr, rng)
    f0_1 = 132.0
    f0_2 = 114.5
    # Realistic faint (hard-to-hear) amplitude buried in ambient + speech
    audio += _synthesize_flyover_signal(t_audio, target1, obs_ecef, t0_utc, f0_1, amplitude=0.075)
    audio += _synthesize_flyover_signal(t_audio, target2, obs_ecef, t0_utc, f0_2, amplitude=0.095)
    audio = np.clip(audio, -0.98, 0.98)

    from .physics import compute_track_cpa
    _, _, t_del1 = compute_track_cpa(target1.timestamps, target1.pos_ecef, target1.vel_ecef, obs_ecef)
    _, _, t_del2 = compute_track_cpa(target2.timestamps, target2.pos_ecef, target2.vel_ecef, obs_ecef)

    return DemoScenario(
        name=f"Colsto Two-Event Benchmark ({int(gap_s)}s offset)",
        mode="two_event",
        samples=audio,
        sample_rate=sr,
        tracks=tracks,
        dem=dem,
        true_lat=true_lat,
        true_lon=true_lon,
        true_elev_m=true_elev,
        recording_start_utc=t0_utc,
        true_event1_audio_s=float(t_del1 - t0_utc),
        true_event2_audio_s=float(t_del2 - t0_utc),
        true_f0_1=f0_1,
        true_f0_2=f0_2,
        known_utc_cpa1=None,
    )
