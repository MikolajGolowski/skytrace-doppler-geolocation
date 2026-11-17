"""Digital Elevation Model (DEM) providers for 3D slant-distance calculation."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np


class DEMProvider(Protocol):
    """Protocol for ground elevation lookup (meters above sea level)."""

    name: str

    def get_elevations(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Return terrain elevation in meters MSL for arrays of lat/lon degrees."""
        ...


class FlatDEM:
    """Constant-elevation terrain model (fallback or flat coastal test)."""

    def __init__(self, elevation_m: float = 0.0) -> None:
        self.elevation_m = float(elevation_m)
        self.name = f"FlatDEM({self.elevation_m:.1f}m)"

    def get_elevations(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        arr = np.asarray(lats, dtype=np.float64)
        return np.full(arr.shape, self.elevation_m, dtype=np.float64)


class SyntheticTerrainDEM:
    """Smooth analytical terrain model for realistic non-flat benchmarks."""

    def __init__(
        self,
        base_m: float = 18.0,
        hill_amp_m: float = 65.0,
        seed_lat: float = 40.87,
        seed_lon: float = -73.50,
    ) -> None:
        self.base_m = base_m
        self.hill_amp_m = hill_amp_m
        self.seed_lat = seed_lat
        self.seed_lon = seed_lon
        self.name = "SyntheticTerrainDEM"

    def get_elevations(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        la = np.asarray(lats, dtype=np.float64)
        lo = np.asarray(lons, dtype=np.float64)
        dlat = (la - self.seed_lat) * 111.0
        dlon = (lo - self.seed_lon) * 85.0
        elev = (
            self.base_m
            + self.hill_amp_m
            * (0.5 + 0.5 * np.sin(dlat * 0.45) * np.cos(dlon * 0.55))
            + 25.0 * np.cos(dlat * 1.1 + dlon * 0.9)
        )
        return np.maximum(elev, 0.0)


class GridDEM:
    """Bilinearly interpolated regular-grid DEM from .npz or ESRI ASCII (.asc) file."""

    def __init__(
        self,
        lats_axis: np.ndarray,
        lons_axis: np.ndarray,
        elev_grid: np.ndarray,
        name: str = "GridDEM",
    ) -> None:
        self.lats = np.asarray(lats_axis, dtype=np.float64)
        self.lons = np.asarray(lons_axis, dtype=np.float64)
        self.grid = np.asarray(elev_grid, dtype=np.float64)
        self.name = name

    @classmethod
    def from_npz(cls, path: str | os.PathLike[str]) -> "GridDEM":
        data = np.load(path)
        return cls(data["lats"], data["lons"], data["elevation"], name=f"GridDEM({Path(path).name})")

    @classmethod
    def from_ascii_grid(cls, path: str | os.PathLike[str]) -> "GridDEM":
        header: dict[str, float] = {}
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(6):
                line = f.readline().strip().split()
                if len(line) >= 2:
                    header[line[0].lower()] = float(line[1])
            nrows = int(header["nrows"])
            ncols = int(header["ncols"])
            xll = header.get("xllcorner", header.get("xllcenter", 0.0))
            yll = header.get("yllcorner", header.get("yllcenter", 0.0))
            cell = header["cellsize"]
            nodata = header.get("nodata_value", -9999.0)
            arr = np.loadtxt(f, dtype=np.float64).reshape((nrows, ncols))
        arr = np.where(arr == nodata, 0.0, arr)
        arr = np.flipud(arr)  # ASCII grid rows go top-to-bottom
        lons = xll + np.arange(ncols) * cell
        lats = yll + np.arange(nrows) * cell
        return cls(lats, lons, arr, name=f"AsciiDEM({Path(path).name})")

    def get_elevations(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        la = np.asarray(lats, dtype=np.float64)
        lo = np.asarray(lons, dtype=np.float64)
        shape = la.shape
        la_f = la.ravel()
        lo_f = lo.ravel()

        iy = np.interp(la_f, self.lats, np.arange(len(self.lats)))
        ix = np.interp(lo_f, self.lons, np.arange(len(self.lons)))

        y0 = np.clip(np.floor(iy).astype(int), 0, len(self.lats) - 2)
        x0 = np.clip(np.floor(ix).astype(int), 0, len(self.lons) - 2)
        y1 = y0 + 1
        x1 = x0 + 1

        wy = np.clip(iy - y0, 0.0, 1.0)
        wx = np.clip(ix - x0, 0.0, 1.0)

        v00 = self.grid[y0, x0]
        v01 = self.grid[y0, x1]
        v10 = self.grid[y1, x0]
        v11 = self.grid[y1, x1]
        out = (
            v00 * (1 - wy) * (1 - wx)
            + v01 * (1 - wy) * wx
            + v10 * wy * (1 - wx)
            + v11 * wy * wx
        )
        return out.reshape(shape)


class OpenMeteoDEM:
    """Network-backed real-world DEM via Open-Meteo Elevation API with disk cache.

    Falls back cleanly to 0.0m if offline or rate-limited.
    """

    def __init__(self, cache_path: str | os.PathLike[str] = ".cache/dem_cache.json") -> None:
        self.cache_path = Path(cache_path)
        self.name = "OpenMeteoDEM"
        self._cache: dict[str, float] = {}
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except Exception:
            pass

    def get_elevations(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        la = np.asarray(lats, dtype=np.float64)
        lo = np.asarray(lons, dtype=np.float64)
        shape = la.shape
        la_f = la.ravel()
        lo_f = lo.ravel()

        out = np.zeros(la_f.shape, dtype=np.float64)
        missing_idx: list[int] = []
        missing_keys: list[str] = []

        for i, (lat, lon) in enumerate(zip(la_f, lo_f)):
            # ~100m cache bucket key
            key = f"{lat:.3f},{lon:.3f}"
            if key in self._cache:
                out[i] = self._cache[key]
            else:
                missing_idx.append(i)
                missing_keys.append(key)

        if missing_idx:
            # Batch up to 100 coordinates per request
            batch_size = 100
            for start in range(0, len(missing_idx), batch_size):
                chunk_idx = missing_idx[start : start + batch_size]
                chunk_keys = missing_keys[start : start + batch_size]
                lat_q = ",".join(f"{la_f[i]:.5f}" for i in chunk_idx)
                lon_q = ",".join(f"{lo_f[i]:.5f}" for i in chunk_idx)
                url = f"https://api.open-meteo.com/v1/elevation?latitude={urllib.parse.quote(lat_q)}&longitude={urllib.parse.quote(lon_q)}"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "AeroDoppler/0.1"})
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                    elevs = payload.get("elevation", [])
                    for idx, key, val in zip(chunk_idx, chunk_keys, elevs):
                        v = float(val) if val is not None else 0.0
                        out[idx] = v
                        self._cache[key] = v
                except Exception:
                    for idx, key in zip(chunk_idx, chunk_keys):
                        out[idx] = 0.0
                        self._cache[key] = 0.0
            self._save_cache()

        return out.reshape(shape)
