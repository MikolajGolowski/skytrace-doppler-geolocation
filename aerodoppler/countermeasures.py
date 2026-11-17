"""Acoustic countermeasure & privacy stress-testing module.

Demonstrates why naive mitigations discussed in Colsto's video fail (high-pass
filtering + MP3 compression + boost recovery, background speech masking) and
what actually disrupts geolocation (time-cut splicing, spectral erasure).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .physics import ideal_doppler_curve
from .spectrogram import SpectrogramData, compute_spectrogram


def apply_highpass_and_compress(
    samples: np.ndarray,
    sr: int,
    cutoff_hz: float = 160.0,
    order: int = 2,
    simulate_lossy_quant: bool = True,
) -> np.ndarray:
    """Apply a gentle/moderate high-pass filter and optional lossy quantization.

    Notice: realistic filter roll-offs (1st/2nd order) leave residual energy in
    the transition band AND fail to erase harmonics (2*f0, 3*f0), allowing
    post-whitening boost to recover the Doppler curve!
    """
    sos = signal.butter(order, min(cutoff_hz, sr * 0.45), btype="highpass", fs=sr, output="sos")
    out = signal.sosfiltfilt(sos, samples)
    if simulate_lossy_quant:
        # Mild spectral quantization noise simulating aggressive codec compression
        rng = np.random.default_rng(7)
        out = out + 0.002 * rng.standard_normal(len(out))
    return out


def apply_masking_speech(
    samples: np.ndarray,
    sr: int,
    snr_penalty_db: float = 10.0,
) -> np.ndarray:
    """Inject loud competing speech-band & broadband noise over the recording."""
    rng = np.random.default_rng(19)
    t = np.arange(len(samples), dtype=np.float64) / sr
    sig_std = max(float(np.std(samples)), 1e-6)
    gain = sig_std * (10.0 ** (snr_penalty_db / 20.0))

    # Multi-tone modulated speech-like mask + broadband noise
    mask = (
        0.55 * np.sin(2 * np.pi * (145.0 + 30.0 * np.sin(2 * np.pi * 1.8 * t)) * t)
        + 0.35 * np.sin(2 * np.pi * (290.0 + 45.0 * np.cos(2 * np.pi * 2.3 * t)) * t)
        + 0.45 * rng.standard_normal(len(samples))
    )
    mask = mask / max(float(np.std(mask)), 1e-6) * gain
    return np.clip(samples + mask, -0.99, 0.99)


def apply_time_cuts(
    samples: np.ndarray,
    sr: int,
    remove_start_s: float,
    remove_end_s: float,
) -> np.ndarray:
    """Splice out a middle segment of audio, breaking the inter-event time offset."""
    i0 = max(0, int(round(remove_start_s * sr)))
    i1 = min(len(samples), int(round(remove_end_s * sr)))
    if i1 <= i0:
        return samples.copy()
    return np.concatenate([samples[:i0], samples[i1:]])


def evaluate_countermeasure(
    samples: np.ndarray,
    sr: int,
    t_cpa_s: float,
    f0_hz: float,
    speed_ms: float,
    d_cpa_m: float,
    method: str,
    param: float = 160.0,
) -> dict:
    """Run a countermeasure on audio and measure whether the Doppler curve survives."""
    if method == "highpass_boost":
        processed = apply_highpass_and_compress(samples, sr, cutoff_hz=param, order=2)
    elif method == "steep_bandstop":
        # Aggressive brickwall high-pass above 500 Hz + codec quantization floor (destroys speech too!)
        sos = signal.butter(12, min(max(param, 500.0), sr * 0.45), btype="highpass", fs=sr, output="sos")
        processed = signal.sosfiltfilt(sos, samples)
        rng = np.random.default_rng(7)
        processed = processed + 1e-4 * rng.standard_normal(len(processed))
    elif method == "speech_masking":
        processed = apply_masking_speech(samples, sr, snr_penalty_db=param)
    elif method == "time_cut":
        mid = len(samples) / sr * 0.5
        processed = apply_time_cuts(samples, sr, mid - param * 0.5, mid + param * 0.5)
    else:
        processed = samples.copy()

    orig_spec: SpectrogramData = compute_spectrogram(samples, sr)
    proc_spec: SpectrogramData = compute_spectrogram(processed, sr)

    t_eval = orig_spec.times[(orig_spec.times >= t_cpa_s - 35.0) & (orig_spec.times <= t_cpa_s + 35.0)]
    if len(t_eval) < 4:
        t_eval = orig_spec.times
    f_curve = ideal_doppler_curve(t_eval, f0_hz, speed_ms, d_cpa_m, t_cpa_s)

    orig_score = orig_spec.eval_curve(t_eval, f_curve, harmonic_weights=(1.0, 0.3, 0.15))
    proc_score = proc_spec.eval_curve(t_eval, f_curve, harmonic_weights=(1.0, 0.3, 0.15))
    retention = (proc_score / max(orig_score, 1e-6)) * 100.0

    # Measure speech-band preservation (120 - 340 Hz energy ratio)
    speech_band_orig = (orig_spec.freqs >= 120.0) & (orig_spec.freqs <= 340.0)
    speech_band_proc = (proc_spec.freqs >= 120.0) & (proc_spec.freqs <= 340.0)
    orig_pwr = float(np.mean(10.0 ** (orig_spec.raw_db[speech_band_orig] / 10.0)))
    proc_pwr = float(np.mean(10.0 ** (proc_spec.raw_db[speech_band_proc] / 10.0)))
    speech_preserved_pct = min(100.0, (proc_pwr / max(orig_pwr, 1e-18)) * 100.0)

    return {
        "method": method,
        "param": param,
        "original_curve_snr": round(orig_score, 3),
        "processed_curve_snr": round(proc_score, 3),
        "curve_retention_pct": round(min(150.0, max(0.0, retention)), 1),
        "speech_preserved_pct": round(speech_preserved_pct, 1),
        "curve_still_detectable": bool(proc_score >= 0.90),
        "spectrogram": proc_spec,
    }
