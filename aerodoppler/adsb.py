"""ADS-B flight track ingestion (adsb.lol readsb traces, CSV, Parquet, JSON, live API)."""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .aircraft_types import PropulsionClass, classify_icao_type
from .physics import geodetic_to_ecef

FEET_TO_METERS = 0.3048
KNOTS_TO_MS = 0.514444


@dataclass
class FlightTrack:
    """3D time-series trajectory for a single aircraft."""

    icao24: str
    callsign: str
    type_code: str
    timestamps: np.ndarray  # Unix seconds, shape (N,)
    lats: np.ndarray        # WGS84 deg, shape (N,)
    lons: np.ndarray        # WGS84 deg, shape (N,)
    altitudes_m: np.ndarray # Meters MSL, shape (N,)
    pos_ecef: np.ndarray = field(init=False, repr=False)
    vel_ecef: np.ndarray = field(init=False, repr=False)
    speeds_ms: np.ndarray = field(init=False, repr=False)
    propulsion_class: PropulsionClass = field(init=False)

    def __post_init__(self) -> None:
        order = np.argsort(self.timestamps)
        self.timestamps = np.asarray(self.timestamps[order], dtype=np.float64)
        self.lats = np.asarray(self.lats[order], dtype=np.float64)
        self.lons = np.asarray(self.lons[order], dtype=np.float64)
        self.altitudes_m = np.asarray(self.altitudes_m[order], dtype=np.float64)

        # Deduplicate identical timestamps
        if len(self.timestamps) > 1:
            keep = np.concatenate(([True], np.diff(self.timestamps) > 1e-3))
            self.timestamps = self.timestamps[keep]
            self.lats = self.lats[keep]
            self.lons = self.lons[keep]
            self.altitudes_m = self.altitudes_m[keep]

        self.propulsion_class = classify_icao_type(self.type_code)
        self.pos_ecef = geodetic_to_ecef(self.lats, self.lons, self.altitudes_m)

        if len(self.timestamps) >= 2:
            vx = np.gradient(self.pos_ecef[:, 0], self.timestamps)
            vy = np.gradient(self.pos_ecef[:, 1], self.timestamps)
            vz = np.gradient(self.pos_ecef[:, 2], self.timestamps)
            self.vel_ecef = np.stack([vx, vy, vz], axis=-1)
        else:
            self.vel_ecef = np.zeros_like(self.pos_ecef)
        self.speeds_ms = np.linalg.norm(self.vel_ecef, axis=1)

    @property
    def min_speed_ms(self) -> float:
        return float(np.min(self.speeds_ms)) if len(self.speeds_ms) else 0.0

    @property
    def max_speed_ms(self) -> float:
        return float(np.max(self.speeds_ms)) if len(self.speeds_ms) else 0.0

    @property
    def median_speed_ms(self) -> float:
        return float(np.median(self.speeds_ms)) if len(self.speeds_ms) else 0.0

    @property
    def start_time(self) -> float:
        return float(self.timestamps[0])

    @property
    def end_time(self) -> float:
        return float(self.timestamps[-1])

    def to_dict(self) -> dict:
        return {
            "icao24": self.icao24,
            "callsign": self.callsign,
            "type_code": self.type_code,
            "propulsion_class": self.propulsion_class,
            "timestamps": self.timestamps.tolist(),
            "lats": self.lats.tolist(),
            "lons": self.lons.tolist(),
            "altitudes_m": self.altitudes_m.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FlightTrack":
        return cls(
            icao24=str(data.get("icao24", data.get("hex", "unknown"))),
            callsign=str(data.get("callsign", "")).strip(),
            type_code=str(data.get("type_code", data.get("type", ""))).strip(),
            timestamps=np.asarray(data["timestamps"], dtype=np.float64),
            lats=np.asarray(data["lats"], dtype=np.float64),
            lons=np.asarray(data["lons"], dtype=np.float64),
            altitudes_m=np.asarray(data["altitudes_m"], dtype=np.float64),
        )


def _split_long_gaps(track: FlightTrack, max_gap_s: float = 180.0) -> list[FlightTrack]:
    """Split a daily aircraft trace into contiguous flight segments on large time gaps."""
    if len(track.timestamps) < 2:
        return []
    gaps = np.where(np.diff(track.timestamps) > max_gap_s)[0]
    if len(gaps) == 0:
        return [track]
    segments: list[FlightTrack] = []
    starts = np.concatenate(([0], gaps + 1))
    ends = np.concatenate((gaps + 1, [len(track.timestamps)]))
    for i, (s, e) in enumerate(zip(starts, ends)):
        if e - s >= 3:
            segments.append(
                FlightTrack(
                    icao24=f"{track.icao24}#{i+1}" if len(starts) > 1 else track.icao24,
                    callsign=track.callsign,
                    type_code=track.type_code,
                    timestamps=track.timestamps[s:e],
                    lats=track.lats[s:e],
                    lons=track.lons[s:e],
                    altitudes_m=track.altitudes_m[s:e],
                )
            )
    return segments


def load_readsb_trace_file(
    path: str | os.PathLike[str],
    bbox: tuple[float, float, float, float] | None = None,
    time_range: tuple[float, float] | None = None,
) -> list[FlightTrack]:
    """Parse an adsb.lol / readsb trace JSON or .json.gz file.

    bbox format: (min_lat, max_lat, min_lon, max_lon)
    time_range format: (start_epoch, end_epoch)
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    base_ts = float(data.get("timestamp", 0.0))
    icao = str(data.get("icao", p.stem.split(".")[0])).strip().lower()
    type_code = str(data.get("t", data.get("type", ""))).strip().upper()
    callsign = str(data.get("r", data.get("flight", ""))).strip()

    raw_trace = data.get("trace", [])
    if not raw_trace:
        return []

    ts_list: list[float] = []
    lat_list: list[float] = []
    lon_list: list[float] = []
    alt_list: list[float] = []

    for row in raw_trace:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        dt, lat, lon, alt_raw = row[0], row[1], row[2], row[3]
        if lat is None or lon is None:
            continue
        t = base_ts + float(dt)
        if time_range and not (time_range[0] <= t <= time_range[1]):
            continue
        lat_f, lon_f = float(lat), float(lon)
        if bbox and not (bbox[0] <= lat_f <= bbox[1] and bbox[2] <= lon_f <= bbox[3]):
            continue
        if isinstance(alt_raw, str) and alt_raw.lower() == "ground":
            alt_m = 0.0
        elif alt_raw is None:
            alt_m = 0.0
        else:
            alt_m = float(alt_raw) * FEET_TO_METERS
        ts_list.append(t)
        lat_list.append(lat_f)
        lon_list.append(lon_f)
        alt_list.append(alt_m)

    if len(ts_list) < 3:
        return []

    track = FlightTrack(
        icao24=icao,
        callsign=callsign,
        type_code=type_code,
        timestamps=np.array(ts_list, dtype=np.float64),
        lats=np.array(lat_list, dtype=np.float64),
        lons=np.array(lon_list, dtype=np.float64),
        altitudes_m=np.array(alt_list, dtype=np.float64),
    )
    return _split_long_gaps(track)


def load_adsb_tabular(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float] | None = None,
    time_range: tuple[float, float] | None = None,
) -> list[FlightTrack]:
    """Load tracks from a DataFrame (CSV / Parquet)."""
    cols = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_id = pick("icao24", "icao", "hex", "flight_id", "id")
    c_ts = pick("timestamp", "time", "ts", "epoch")
    c_lat = pick("lat", "latitude")
    c_lon = pick("lon", "lng", "longitude")
    c_alt_m = pick("alt_m", "altitude_m", "alt")
    c_alt_ft = pick("alt_ft", "altitude_ft", "alt_baro", "alt_geom")
    c_type = pick("type_code", "type", "aircraft_type", "icao_type")
    c_call = pick("callsign", "flight", "registration")

    if not all([c_id, c_ts, c_lat, c_lon]):
        raise ValueError("Tabular ADS-B data must contain id/icao24, timestamp, lat, lon columns.")

    work = df.copy()
    if pd.api.types.is_string_dtype(work[c_ts]) or pd.api.types.is_datetime64_any_dtype(work[c_ts]):
        parsed = pd.to_datetime(work[c_ts], errors="coerce", utc=True)
        if parsed.notna().sum() > len(work) * 0.5:
            work["_ts"] = parsed.astype("int64") / 1e9
        else:
            work["_ts"] = pd.to_numeric(work[c_ts], errors="coerce")
    else:
        work["_ts"] = pd.to_numeric(work[c_ts], errors="coerce")

    work = work.dropna(subset=[c_id, "_ts", c_lat, c_lon])
    if time_range:
        work = work[(work["_ts"] >= time_range[0]) & (work["_ts"] <= time_range[1])]
    if bbox:
        work = work[
            (work[c_lat] >= bbox[0])
            & (work[c_lat] <= bbox[1])
            & (work[c_lon] >= bbox[2])
            & (work[c_lon] <= bbox[3])
        ]

    if c_alt_m:
        work["_alt_m"] = pd.to_numeric(work[c_alt_m], errors="coerce").fillna(0.0)
    elif c_alt_ft:
        work["_alt_m"] = pd.to_numeric(work[c_alt_ft], errors="coerce").fillna(0.0) * FEET_TO_METERS
    else:
        work["_alt_m"] = 0.0

    tracks: list[FlightTrack] = []
    for icao, grp in work.groupby(c_id):
        if len(grp) < 3:
            continue
        type_val = str(grp[c_type].dropna().iloc[0]) if c_type and grp[c_type].notna().any() else ""
        call_val = str(grp[c_call].dropna().iloc[0]) if c_call and grp[c_call].notna().any() else str(icao)
        ft = FlightTrack(
            icao24=str(icao).strip(),
            callsign=call_val.strip(),
            type_code=type_val.strip().upper(),
            timestamps=grp["_ts"].to_numpy(dtype=np.float64),
            lats=grp[c_lat].to_numpy(dtype=np.float64),
            lons=grp[c_lon].to_numpy(dtype=np.float64),
            altitudes_m=grp["_alt_m"].to_numpy(dtype=np.float64),
        )
        tracks.extend(_split_long_gaps(ft))
    return tracks


def load_adsb_path(
    path: str | os.PathLike[str],
    bbox: tuple[float, float, float, float] | None = None,
    time_range: tuple[float, float] | None = None,
) -> list[FlightTrack]:
    """Load ADS-B tracks from file (JSON, readsb .json/.json.gz, CSV, Parquet) or directory."""
    p = Path(path)
    if p.is_dir():
        tracks: list[FlightTrack] = []
        for child in sorted(p.rglob("*")):
            if child.is_file() and (
                child.name.endswith(".json")
                or child.name.endswith(".json.gz")
            ):
                tracks.extend(load_readsb_trace_file(child, bbox=bbox, time_range=time_range))
        return tracks

    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".csv"):
        return load_adsb_tabular(pd.read_csv(p), bbox=bbox, time_range=time_range)
    if suffixes.endswith(".parquet") or suffixes.endswith(".pq"):
        return load_adsb_tabular(pd.read_parquet(p), bbox=bbox, time_range=time_range)
    if suffixes.endswith(".json") or suffixes.endswith(".json.gz"):
        opener = gzip.open if suffixes.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8", errors="replace") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "tracks" in payload:
            out: list[FlightTrack] = []
            for item in payload["tracks"]:
                ft = FlightTrack.from_dict(item)
                if bbox:
                    mask = (
                        (ft.lats >= bbox[0])
                        & (ft.lats <= bbox[1])
                        & (ft.lons >= bbox[2])
                        & (ft.lons <= bbox[3])
                    )
                    if np.sum(mask) < 3:
                        continue
                out.append(ft)
            return out
        if isinstance(payload, list):
            return [FlightTrack.from_dict(item) for item in payload]
        return load_readsb_trace_file(p, bbox=bbox, time_range=time_range)
    raise ValueError(f"Unsupported ADS-B file format: {p}")


def save_adsb_json(tracks: Iterable[FlightTrack], path: str | os.PathLike[str]) -> None:
    """Save flight tracks to unified JSON format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tracks": [t.to_dict() for t in tracks]}
    p.write_text(json.dumps(payload), encoding="utf-8")


def fetch_adsblol_live(lat: float, lon: float, dist_nm: int = 25) -> list[dict]:
    """Fetch live aircraft positions around (lat, lon) from public adsb.lol v2 API."""
    url = f"https://api.adsb.lol/v2/lat/{lat:.4f}/lon/{lon:.4f}/dist/{int(dist_nm)}"
    req = urllib.request.Request(url, headers={"User-Agent": "AeroDoppler/0.1"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("ac", [])
