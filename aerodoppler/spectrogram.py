"""High-resolution audio spectrogram extraction, whitening, and Doppler curve fitting."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, optimize, signal

from .physics import DEFAULT_SPEED_OF_SOUND, ideal_doppler_curve


@dataclass
class SpectrogramData:
    """Container for zoomed time-frequency spectrogram matrices."""

    times: np.ndarray       # shape (T,), seconds from clip start
    freqs: np.ndarray       # shape (F,), Hz
    raw_db: np.ndarray      # shape (F, T), dB power
    whitened: np.ndarray    # shape (F, T), robust Z-score >= 0
    smoothed: np.ndarray    # shape (F, T), slightly smoothed along freq for robust ridge scoring
    sample_rate: int
    duration_s: float

    def eval_curve(
        self,
        t_curve: np.ndarray,
        f_curve: np.ndarray,
        harmonic_weights: tuple[float, ...] = (1.0, 0.3),
        use_whitened: bool = True,
        use_smoothed: bool = False,
    ) -> float:
        """Sample and sum (mean-normalized) spectrogram brightness along a curve."""
        if use_smoothed:
            mat = self.smoothed
        else:
            mat = self.whitened if use_whitened else self.raw_db
        valid = (
            np.isfinite(t_curve)
            & np.isfinite(f_curve)
            & (t_curve >= self.times[0])
            & (t_curve <= self.times[-1])
        )
        if not np.any(valid):
            return 0.0

        tc = t_curve[valid]
        fc = f_curve[valid]
        dt = float(self.times[1] - self.times[0]) if len(self.times) > 1 else 1.0
        df = float(self.freqs[1] - self.freqs[0]) if len(self.freqs) > 1 else 1.0

        tx = (tc - self.times[0]) / dt
        tx0 = np.clip(np.floor(tx).astype(int), 0, len(self.times) - 2)
        wx = np.clip(tx - tx0, 0.0, 1.0)

        total_score = 0.0
        for h_idx, w_h in enumerate(harmonic_weights, start=1):
            if w_h <= 0.0:
                continue
            fh = fc * h_idx
            in_band = (fh >= self.freqs[0]) & (fh <= self.freqs[-1])
            if not np.any(in_band):
                continue
            fy = (fh[in_band] - self.freqs[0]) / df
            fy0 = np.clip(np.floor(fy).astype(int), 0, len(self.freqs) - 2)
            wy = np.clip(fy - fy0, 0.0, 1.0)

            x0 = tx0[in_band]
            wx_b = wx[in_band]
            v00 = mat[fy0, x0]
            v01 = mat[fy0, x0 + 1]
            v10 = mat[fy0 + 1, x0]
            v11 = mat[fy0 + 1, x0 + 1]
            vals = (
                v00 * (1 - wy) * (1 - wx_b)
                + v01 * (1 - wy) * wx_b
                + v10 * wy * (1 - wx_b)
                + v11 * wy * wx_b
            )
            total_score += w_h * float(np.mean(vals))
        return float(total_score)


@dataclass
class DetectedDopplerEvent:
    """Measured Doppler event parameters from spectrogram fitting."""

    t_cpa_s: float
    f0_hz: float
    speed_ms: float
    d_cpa_m: float
    score: float
    speed_min_ms: float
    speed_max_ms: float
    d_min_m: float
    d_max_m: float
    t_tolerance_s: float = 5.0
    f0_tolerance_hz: float = 4.0

    def to_dict(self) -> dict:
        return {
            "t_cpa_s": round(self.t_cpa_s, 2),
            "f0_hz": round(self.f0_hz, 2),
            "speed_ms": round(self.speed_ms, 2),
            "d_cpa_m": round(self.d_cpa_m, 1),
            "score": round(self.score, 3),
            "speed_min_ms": round(self.speed_min_ms, 1),
            "speed_max_ms": round(self.speed_max_ms, 1),
            "d_min_m": round(self.d_min_m, 1),
            "d_max_m": round(self.d_max_m, 1),
            "t_tolerance_s": round(self.t_tolerance_s, 2),
            "f0_tolerance_hz": round(self.f0_tolerance_hz, 2),
        }


def load_audio(
    path: str | os.PathLike[str],
    target_sr: int | None = 2000,
) -> tuple[np.ndarray, int]:
    """Load mono float64 audio from WAV/FLAC/OGG or via ffmpeg for MP3/MP4/MKV."""
    p = Path(path)
    try:
        data, sr = sf.read(str(p), always_2d=True, dtype="float64")
        mono = np.mean(data, axis=1)
    except Exception as exc:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                f"Cannot read {p.name} directly with soundfile ({exc}) and ffmpeg is not installed."
            ) from exc
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cmd = [ffmpeg, "-y", "-i", str(p), "-ac", "1", "-ar", str(target_sr or 2000), tmp_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            data, sr = sf.read(tmp_path, always_2d=False, dtype="float64")
            mono = np.asarray(data, dtype=np.float64)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if target_sr is not None and sr > target_sr:
        g = gcd(int(sr), int(target_sr))
        up = int(target_sr) // g
        down = int(sr) // g
        mono = signal.resample_poly(mono, up, down)
        sr = int(target_sr)
    return mono, int(sr)


def boost_band_audio_wav_bytes(
    samples: np.ndarray,
    sr: int,
    low_hz: float = 55.0,
    high_hz: float = 350.0,
) -> bytes:
    """Band-pass boost aircraft frequencies and normalize for audible playback."""
    nyq = sr * 0.5
    hi = min(high_hz, nyq * 0.92)
    lo = max(low_hz, 15.0)
    if lo < hi:
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
        filtered = signal.sosfiltfilt(sos, samples)
    else:
        filtered = samples.copy()
    peak = float(np.max(np.abs(filtered)))
    if peak > 1e-9:
        filtered = 0.85 * filtered / peak
    buf = io.BytesIO()
    sf.write(buf, filtered.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def compute_spectrogram(
    samples: np.ndarray,
    sr: int,
    fmin: float = 40.0,
    fmax: float = 360.0,
    win_sec: float = 4.0,
    hop_sec: float = 0.5,
    zero_pad_factor: int = 2,
) -> SpectrogramData:
    """Compute zoomed, noise-whitened STFT spectrogram optimized for aircraft Doppler curves."""
    nperseg = max(256, int(round(win_sec * sr)))
    nperseg = min(nperseg, len(samples))
    hop = max(16, int(round(hop_sec * sr)))
    noverlap = max(0, nperseg - hop)
    nfft = int(2 ** np.ceil(np.log2(nperseg * zero_pad_factor)))

    freqs, times, Zxx = signal.stft(
        samples,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    power = np.abs(Zxx) ** 2

    band = (freqs >= fmin) & (freqs <= fmax)
    freqs_z = freqs[band]
    power_z = power[band, :]
    raw_db = 10.0 * np.log10(np.maximum(power_z, 1e-18))

    # Two-way robust background equalization:
    # 1) Remove stationary tonal lines (50/60 Hz hums) via per-bin 35th percentile across time.
    bin_floor = np.percentile(raw_db, 35, axis=1, keepdims=True)
    detoned = raw_db - bin_floor

    # 2) Remove broadband transients (wind gusts, speech) via per-frame median across frequency.
    frame_floor = np.median(detoned, axis=0, keepdims=True)
    cleaned = detoned - frame_floor

    # 3) Robust Z-score using MAD, clipped for clean contrast
    med = float(np.median(cleaned))
    mad = float(np.median(np.abs(cleaned - med))) * 1.4826
    scale = max(mad, 0.5)
    whitened = np.clip((cleaned - med) / scale, 0.0, 10.0)

    df = float(freqs_z[1] - freqs_z[0]) if len(freqs_z) > 1 else 1.0
    sigma_bins = max(0.8, 1.0 / df)
    smoothed = ndimage.gaussian_filter1d(whitened, sigma=sigma_bins, axis=0)

    return SpectrogramData(
        times=times,
        freqs=freqs_z,
        raw_db=raw_db,
        whitened=whitened,
        smoothed=smoothed,
        sample_rate=sr,
        duration_s=float(len(samples) / sr),
    )


def _vectorized_f0_sweep(
    spec: SpectrogramData,
    t_idx: np.ndarray,
    f_unit: np.ndarray,
    f0_grid: np.ndarray,
    use_smoothed: bool = True,
) -> tuple[float, float]:
    """Vectorized sweep over f0_grid at integer time indices `t_idx`."""
    mat = spec.smoothed if use_smoothed else spec.whitened
    df = float(spec.freqs[1] - spec.freqs[0])
    f_min = float(spec.freqs[0])
    n_freqs = len(spec.freqs)

    # Shape: (len(f0_grid), len(t_idx))
    f_mat = f0_grid[:, None] * f_unit[None, :]
    fy = np.rint((f_mat - f_min) / df).astype(np.int64)
    in_band = (fy >= 0) & (fy < n_freqs)
    fy_c = np.clip(fy, 0, n_freqs - 1)
    vals = np.where(in_band, mat[fy_c, t_idx[None, :]], 0.0)
    counts = np.maximum(np.sum(in_band, axis=1), 1)
    scores = np.sum(vals, axis=1) / counts
    best_i = int(np.argmax(scores))
    return float(scores[best_i]), float(f0_grid[best_i])


def fit_ideal_doppler(
    spec: SpectrogramData,
    t_center_hint: float,
    t_window_s: float = 44.0,
    f0_range: tuple[float, float] = (65.0, 240.0),
    v_range: tuple[float, float] = (40.0, 140.0),
    d_range: tuple[float, float] = (300.0, 5000.0),
    c: float = DEFAULT_SPEED_OF_SOUND,
) -> DetectedDopplerEvent:
    """Fit ideal constant-velocity Doppler curve to a spectrogram region around `t_center_hint`."""
    half = t_window_s * 0.5
    idx_mask = np.where((spec.times >= t_center_hint - half) & (spec.times <= t_center_hint + half))[0]
    if len(idx_mask) < 5:
        idx_mask = np.arange(len(spec.times))
    t_win = spec.times[idx_mask]

    # Coarse grid search over (t_cpa, f0, v, d_cpa) with 1 Hz f0 steps (vectorized)
    t_grid = np.linspace(
        max(spec.times[0] + 4.0, t_center_hint - 10.0),
        min(spec.times[-1] - 4.0, t_center_hint + 10.0),
        9,
    )
    f0_grid = np.arange(f0_range[0], f0_range[1] + 0.5, 1.0, dtype=np.float64)
    v_grid = np.linspace(v_range[0], v_range[1], 8)
    d_grid = np.geomspace(d_range[0], d_range[1], 7)

    best_score = -1.0
    best_params = (t_center_hint, float(np.mean(f0_range)), float(np.mean(v_range)), 1500.0)

    for v in v_grid:
        for d in d_grid:
            for tc in t_grid:
                f_unit = ideal_doppler_curve(t_win, 1.0, float(v), float(d), float(tc), c=c)
                sc, f0_best = _vectorized_f0_sweep(spec, idx_mask, f_unit, f0_grid, use_smoothed=True)
                if sc > best_score:
                    best_score = sc
                    best_params = (float(tc), float(f0_best), float(v), float(d))

    # Fine Nelder-Mead local optimization around coarse winner
    def neg_objective(x: np.ndarray) -> float:
        tc, f0, v, d = x
        if not (spec.times[0] <= tc <= spec.times[-1]):
            return 0.0
        if not (f0_range[0] <= f0 <= f0_range[1] and v_range[0] <= v <= v_range[1] and d_range[0] <= d <= d_range[1]):
            return 0.0
        fc = ideal_doppler_curve(t_win, f0, v, d, tc, c=c)
        return -spec.eval_curve(t_win, fc, harmonic_weights=(1.0, 0.35), use_smoothed=True)

    res = optimize.minimize(
        neg_objective,
        x0=np.array(best_params, dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": 180, "xatol": 0.15, "fatol": 1e-3},
    )
    tc_opt, f0_opt, v_opt, d_opt = [float(val) for val in res.x]
    final_score = -float(res.fun) if res.fun < 0 else best_score

    # Conservative uncertainty ranges (matching Colsto's generous range filters)
    v_min = max(25.0, v_opt * 0.72)
    v_max = min(220.0, v_opt * 1.28)
    d_min = max(100.0, d_opt * 0.50)
    d_max = min(12000.0, d_opt * 1.75)

    return DetectedDopplerEvent(
        t_cpa_s=tc_opt,
        f0_hz=f0_opt,
        speed_ms=v_opt,
        d_cpa_m=d_opt,
        score=final_score,
        speed_min_ms=v_min,
        speed_max_ms=v_max,
        d_min_m=d_min,
        d_max_m=d_max,
        t_tolerance_s=6.0,
        f0_tolerance_hz=4.0,
    )


def detect_doppler_events(
    spec: SpectrogramData,
    max_events: int = 3,
    min_separation_s: float = 40.0,
    f0_range: tuple[float, float] = (65.0, 240.0),
    c: float = DEFAULT_SPEED_OF_SOUND,
) -> list[DetectedDopplerEvent]:
    """Automatically scan a spectrogram to discover and fit aircraft flyover Doppler curves."""
    if len(spec.times) < 8:
        return []

    dt = float(spec.times[1] - spec.times[0])
    half_frames = max(3, int(round(16.0 / dt)))
    scores_t = np.zeros(len(spec.times), dtype=np.float64)

    probe_f0s = np.arange(f0_range[0], f0_range[1] + 0.5, 1.5, dtype=np.float64)
    for i in range(half_frames, len(spec.times) - half_frames):
        t_c = float(spec.times[i])
        idx_seg = np.arange(i - half_frames, i + half_frames + 1)
        t_seg = spec.times[idx_seg]
        f_unit = ideal_doppler_curve(t_seg, 1.0, 75.0, 1200.0, t_c, c=c)
        sc, _ = _vectorized_f0_sweep(spec, idx_seg, f_unit, probe_f0s, use_smoothed=True)
        scores_t[i] = sc

    candidate_times: list[float] = []
    order = np.argsort(scores_t)[::-1]
    for idx in order:
        if scores_t[idx] <= 0.45:
            break
        tc = float(spec.times[idx])
        if all(abs(tc - prev) >= min_separation_s for prev in candidate_times):
            candidate_times.append(tc)
        if len(candidate_times) >= max_events:
            break

    events: list[DetectedDopplerEvent] = []
    for tc in sorted(candidate_times):
        ev = fit_ideal_doppler(spec, t_center_hint=tc, f0_range=f0_range, c=c)
        events.append(ev)
    return events
