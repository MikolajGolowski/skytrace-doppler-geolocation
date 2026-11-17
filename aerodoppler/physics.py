"""3D geodesic coordinate transforms and retarded-time acoustic Doppler physics."""

from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)

DEFAULT_SPEED_OF_SOUND = 343.0  # m/s (~19.65 C)


def speed_of_sound_from_temp(temp_c: float = 15.0) -> float:
    """Return speed of sound in dry air (m/s) for ambient temperature in Celsius."""
    return float(331.3 * np.sqrt(1.0 + temp_c / 273.15))


def geodetic_to_ecef(
    lat_deg: np.ndarray | float,
    lon_deg: np.ndarray | float,
    alt_m: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Convert WGS84 geodetic (lat, lon, alt_m) to ECEF XYZ coordinates in meters.

    Returns array of shape (..., 3).
    """
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    alt = np.asarray(alt_m, dtype=np.float64)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)

    x = (n + alt) * cos_lat * np.cos(lon)
    y = (n + alt) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ecef_to_geodetic(ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF XYZ (..., 3) in meters to WGS84 (lat_deg, lon_deg, alt_m)."""
    xyz = np.asarray(ecef, dtype=np.float64)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    p = np.sqrt(x**2 + y**2)
    lon = np.arctan2(y, x)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    lat = np.arctan2(
        z + WGS84_EP2 * WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    alt = p / np.maximum(np.cos(lat), 1e-12) - n
    return np.rad2deg(lat), np.rad2deg(lon), alt


def _enu_rotation(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    slat, clat = np.sin(lat0), np.cos(lat0)
    slon, clon = np.sin(lon0), np.cos(lon0)
    return np.array(
        [
            [-slon, clon, 0.0],
            [-slat * clon, -slat * slon, clat],
            [clat * clon, clat * slon, slat],
        ],
        dtype=np.float64,
    )


def geodetic_to_enu(
    lat_deg: np.ndarray | float,
    lon_deg: np.ndarray | float,
    alt_m: np.ndarray | float,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float = 0.0,
) -> np.ndarray:
    """Convert geodetic coordinates to local East-North-Up meters relative to origin."""
    p = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    p0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    rot = _enu_rotation(lat0_deg, lon0_deg)
    return (p - p0) @ rot.T


def enu_to_geodetic(
    enu: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert local ENU meters relative to origin back to WGS84 (lat, lon, alt)."""
    p0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    rot = _enu_rotation(lat0_deg, lon0_deg)
    ecef = np.asarray(enu, dtype=np.float64) @ rot + p0
    return ecef_to_geodetic(ecef)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle horizontal distance in meters between two WGS84 points."""
    r = 6371000.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def ideal_doppler_curve(
    t_heard: np.ndarray,
    f0: float,
    speed_ms: float,
    d_cpa_m: float,
    t_cpa_delayed: float,
    c: float = DEFAULT_SPEED_OF_SOUND,
) -> np.ndarray:
    """Compute exact retarded-time Doppler curve for constant linear motion.

    Parameters
    ----------
    t_heard : array of observation timestamps (seconds)
    f0 : fundamental emitted frequency (Hz)
    speed_ms : aircraft speed (m/s), must satisfy 0 <= speed_ms < c
    d_cpa_m : 3D slant distance at Closest Point of Approach (meters)
    t_cpa_delayed : heard timestamp of the Closest Point of Approach (seconds)
    c : speed of sound (m/s)
    """
    t_h = np.asarray(t_heard, dtype=np.float64)
    v = float(np.clip(speed_ms, 0.0, c * 0.99))
    d_cpa = max(float(d_cpa_m), 1.0)

    tau_h = t_h - t_cpa_delayed
    cap_t = tau_h + d_cpa / c
    beta2 = (v / c) ** 2

    disc = np.maximum(v**2 * cap_t**2 + (1.0 - beta2) * d_cpa**2, 0.0)
    tau_e = (c**2 * cap_t - c * np.sqrt(disc)) / (c**2 - v**2)

    dist = c * (cap_t - tau_e)
    dist = np.maximum(dist, 1e-6)
    v_radial = -(v**2 * tau_e) / dist
    return f0 * (c / (c - v_radial))


def compute_track_cpa(
    track_t: np.ndarray,
    track_pos_ecef: np.ndarray,
    track_vel_ecef: np.ndarray,
    obs_pos_ecef: np.ndarray,
    c: float = DEFAULT_SPEED_OF_SOUND,
) -> tuple[float, float, float]:
    """Compute (t_cpa, d_cpa_m, t_cpa_delayed) between a 3D trajectory and observer.

    Uses discrete minimum followed by sub-second linear projection refinement.
    """
    diff = track_pos_ecef - obs_pos_ecef
    dists = np.linalg.norm(diff, axis=1)
    idx = int(np.argmin(dists))

    t_cpa = float(track_t[idx])
    p_cpa = track_pos_ecef[idx]
    v_cpa = track_vel_ecef[idx]

    v_sq = float(np.dot(v_cpa, v_cpa))
    if v_sq > 1.0 and 0 < idx < len(track_t) - 1:
        dt_step = -float(np.dot(p_cpa - obs_pos_ecef, v_cpa)) / v_sq
        max_dt = 0.5 * float(track_t[min(idx + 1, len(track_t) - 1)] - track_t[max(idx - 1, 0)])
        dt_step = float(np.clip(dt_step, -max_dt, max_dt))
        t_cpa += dt_step
        p_cpa = p_cpa + v_cpa * dt_step

    d_cpa = float(np.linalg.norm(p_cpa - obs_pos_ecef))
    t_cpa_delayed = t_cpa + d_cpa / c
    return t_cpa, d_cpa, t_cpa_delayed


def model_doppler_retarded(
    track_t: np.ndarray,
    track_pos_ecef: np.ndarray,
    track_vel_ecef: np.ndarray,
    obs_pos_ecef: np.ndarray,
    t_heard_eval: np.ndarray,
    f0: float,
    c: float = DEFAULT_SPEED_OF_SOUND,
) -> tuple[np.ndarray, np.ndarray]:
    """Model the exact Doppler curve heard at `t_heard_eval` using retarded emission time.

    Returns
    -------
    freqs_hz : array of modeled frequencies at `t_heard_eval` (NaN where out of track bounds)
    valid_mask : boolean mask indicating where `t_heard_eval` falls within track arrival span
    """
    diff = track_pos_ecef - obs_pos_ecef
    dists = np.linalg.norm(diff, axis=1)
    t_heard_track = track_t + dists / c

    # Ensure strictly monotonic arrival times (subsonic flight guarantees this, guard against noisy timestamps)
    mono_mask = np.concatenate(([True], np.diff(t_heard_track) > 1e-4))
    th_clean = t_heard_track[mono_mask]
    te_clean = track_t[mono_mask]
    pos_clean = track_pos_ecef[mono_mask]
    vel_clean = track_vel_ecef[mono_mask]

    t_h = np.asarray(t_heard_eval, dtype=np.float64)
    valid = (t_h >= th_clean[0]) & (t_h <= th_clean[-1])

    freqs = np.full_like(t_h, np.nan, dtype=np.float64)
    if not np.any(valid):
        return freqs, valid

    t_e = np.interp(t_h[valid], th_clean, te_clean)

    px = np.interp(t_e, te_clean, pos_clean[:, 0])
    py = np.interp(t_e, te_clean, pos_clean[:, 1])
    pz = np.interp(t_e, te_clean, pos_clean[:, 2])
    vx = np.interp(t_e, te_clean, vel_clean[:, 0])
    vy = np.interp(t_e, te_clean, vel_clean[:, 1])
    vz = np.interp(t_e, te_clean, vel_clean[:, 2])

    p_emit = np.stack([px, py, pz], axis=-1)
    v_emit = np.stack([vx, vy, vz], axis=-1)

    vec_to_obs = obs_pos_ecef - p_emit
    d_emit = np.maximum(np.linalg.norm(vec_to_obs, axis=1), 1e-6)
    u_to_obs = vec_to_obs / d_emit[:, None]

    v_radial = np.sum(v_emit * u_to_obs, axis=1)
    v_radial = np.clip(v_radial, -c * 0.95, c * 0.95)
    freqs[valid] = f0 * (c / (c - v_radial))
    return freqs, valid


def model_doppler_naive(
    track_t: np.ndarray,
    track_pos_ecef: np.ndarray,
    track_vel_ecef: np.ndarray,
    obs_pos_ecef: np.ndarray,
    t_heard_eval: np.ndarray,
    f0: float,
    c: float = DEFAULT_SPEED_OF_SOUND,
) -> tuple[np.ndarray, np.ndarray]:
    """Unretarded (naive) Doppler evaluation at t_obs without sound propagation delay.

    Provided specifically to demonstrate why ignoring emission-time retardation
    produces distorted Doppler curves (as highlighted in the video).
    """
    t_h = np.asarray(t_heard_eval, dtype=np.float64)
    valid = (t_h >= track_t[0]) & (t_h <= track_t[-1])
    freqs = np.full_like(t_h, np.nan, dtype=np.float64)
    if not np.any(valid):
        return freqs, valid

    t_q = t_h[valid]
    px = np.interp(t_q, track_t, track_pos_ecef[:, 0])
    py = np.interp(t_q, track_t, track_pos_ecef[:, 1])
    pz = np.interp(t_q, track_t, track_pos_ecef[:, 2])
    vx = np.interp(t_q, track_t, track_vel_ecef[:, 0])
    vy = np.interp(t_q, track_t, track_vel_ecef[:, 1])
    vz = np.interp(t_q, track_t, track_vel_ecef[:, 2])

    p_cur = np.stack([px, py, pz], axis=-1)
    v_cur = np.stack([vx, vy, vz], axis=-1)
    vec_to_obs = obs_pos_ecef - p_cur
    d_cur = np.maximum(np.linalg.norm(vec_to_obs, axis=1), 1e-6)
    u_to_obs = vec_to_obs / d_cur[:, None]
    v_radial = np.clip(np.sum(v_cur * u_to_obs, axis=1), -c * 0.95, c * 0.95)
    freqs[valid] = f0 * (c / (c - v_radial))
    return freqs, valid
