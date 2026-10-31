"""Satellite fire detections turned into what a city's air feels: fire power by distance and direction.

Two passes over data/firms:
1. Permanent hotspots. Gas flares, steel works, refineries and burning coal seams show up as "fires" in the
   same place month after month, all year round. They don't change from day to day, so they can't explain a
   smoke episode, and they would drown the real signal near oil fields. The map is cut into cells of 0.02
   degrees (about 2 km); a cell with detections in at least 35% of the months on record, spread over at least
   9 of the 12 calendar months, is marked permanent and ignored. The second condition keeps places that burn
   every dry season (savanna, crop stubble), which can reach 35% of months but never burn all year.
   Saved to data/clean/fire_static_cells.parquet.
2. Per city. Every remaining detection within 1,000 km of a city (in the city's fire region) is placed in a
   distance ring (0-100, 100-300, 300-600, 600-1,000 km) and one of 8 compass sectors seen from the city,
   and its fire radiative power (FRP, megawatts; how much heat, and roughly how much smoke, a fire gives off)
   is summed per 6-hour period. Low-confidence detections are left out.
   Saved to data/clean/fires_by_city.parquet: city, period_utc, ring_km, sector, frp, n.

Sector 0 is north and they go clockwise in 45 degree steps, so sector 2 holds fires to the east of the city.
Later, the sector a fire is in is compared with the wind direction to tell whether its smoke blows towards
the city (see features.py).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aqcast.settings import Settings
from aqcast.storage import as_utc, read_frame, write_frame

log = logging.getLogger(__name__)

CELL_DEG = 0.02
STATIC_SHARE = 0.35          # share of months with detections that marks a cell as permanent...
STATIC_CALENDAR_MONTHS = 9   # ...if those months also cover most of the year
MONTH_BASE = 2000 * 12       # month numbers are stored as months since Jan 2000 (a multiple of 12)
RINGS_KM = (0, 100, 300, 600, 1000)
SECTORS = 8
PERIOD = "6h"
COLUMNS = ["latitude", "longitude", "frp", "confidence", "detected_utc"]
EARTH_KM = 6371.0


def cell_key(lat, lon) -> np.ndarray:
    """One integer per 0.02 degree cell."""
    i = np.floor(np.asarray(lat, dtype=float) / CELL_DEG).astype(np.int64) + 5000
    j = np.floor(np.asarray(lon, dtype=float) / CELL_DEG).astype(np.int64) + 10000
    return i * 20000 + j


def month_index(ts: pd.Series) -> np.ndarray:
    t = pd.DatetimeIndex(ts)
    return (t.year * 12 + t.month - 1).to_numpy().astype(np.int64) - MONTH_BASE


def distance_bearing(lat0: float, lon0: float, lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle distance (km) and initial bearing (degrees, 0 = north, clockwise) from a city to points."""
    p0, l0 = np.radians(lat0), np.radians(lon0)
    p, l = np.radians(np.asarray(lat, dtype=float)), np.radians(np.asarray(lon, dtype=float))
    dl = l - l0
    a = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    dist = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    y = np.sin(dl) * np.cos(p)
    x = np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(dl)
    bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return dist, bearing


def sector_of(bearing) -> np.ndarray:
    width = 360 / SECTORS
    return (np.floor(((np.asarray(bearing) + width / 2) % 360) / width)).astype(np.int64)


def fire_files(settings: Settings, region: str | None = None) -> list[Path]:
    base = settings.data_dir / "firms"
    pattern = f"{region}/*/*.parquet" if region else "*/*/*.parquet"
    return sorted(base.glob(pattern))


def read_fires(path: Path) -> pd.DataFrame:
    df = read_frame(path, columns=COLUMNS)
    df["detected_utc"] = as_utc(df["detected_utc"])
    return df.dropna(subset=["latitude", "longitude", "frp", "detected_utc"])


# ------------------------------------------------------------------------------------------- pass 1
def static_cells(frames) -> pd.DataFrame:
    """Cells with detections in at least STATIC_SHARE of the months on record. `frames` yields detections."""
    pairs, n_seen, months = [], 0, set()
    for df in frames:
        if df.empty:
            continue
        m = month_index(df["detected_utc"])
        months.update(np.unique(m).tolist())
        pairs.append(np.unique(cell_key(df["latitude"], df["longitude"]) * 10000 + m))
        n_seen += 1
        if len(pairs) >= 200:                       # keep memory low: merge as we go
            pairs = [np.unique(np.concatenate(pairs))]
    if not pairs:
        return pd.DataFrame(columns=["cell", "months", "lat", "lon"])
    u = np.unique(np.concatenate(pairs))
    cell_of, month_of = u // 10000, u % 10000
    cells, counts = np.unique(cell_of, return_counts=True)
    # Distinct calendar months (Jan..Dec) per cell.
    cal_cells, cal_counts = np.unique(np.unique(cell_of * 12 + month_of % 12) // 12, return_counts=True)
    calendar = pd.Series(cal_counts, index=cal_cells).reindex(cells).to_numpy()
    need = max(3, int(np.ceil(STATIC_SHARE * len(months))))
    keep = (counts >= need) & (calendar >= STATIC_CALENDAR_MONTHS)
    cells, counts = cells[keep], counts[keep]
    i, j = cells // 20000 - 5000, cells % 20000 - 10000
    return pd.DataFrame({"cell": cells, "months": counts,
                         "lat": (i + 0.5) * CELL_DEG, "lon": (j + 0.5) * CELL_DEG})


# ------------------------------------------------------------------------------------------- pass 2
def city_rings(df: pd.DataFrame, city_slug: str, lat: float, lon: float) -> pd.DataFrame:
    """Sum FRP per 6-hour period, distance ring and compass sector for one city."""
    dist, bearing = distance_bearing(lat, lon, df["latitude"], df["longitude"])
    near = dist < RINGS_KM[-1]
    if not near.any():
        return pd.DataFrame(columns=["city", "period_utc", "ring_km", "sector", "frp", "n"])
    # Plain UTC datetime64 values (tz-aware columns turn into slow Python objects in to_numpy on pandas 2).
    when = pd.DatetimeIndex(df["detected_utc"]).tz_convert("UTC").tz_localize(None).to_numpy()
    sub = pd.DataFrame({
        "period_utc": when[near],
        "ring_km": np.asarray(RINGS_KM[:-1])[np.digitize(dist[near], RINGS_KM[1:-1])],
        "sector": sector_of(bearing[near]),
        "frp": df["frp"].to_numpy(dtype=float)[near],
    })
    sub["period_utc"] = as_utc(sub["period_utc"].dt.floor(PERIOD))
    out = sub.groupby(["period_utc", "ring_km", "sector"]).agg(frp=("frp", "sum"), n=("frp", "size")).reset_index()
    out.insert(0, "city", city_slug)
    return out


def keep_real_fires(df: pd.DataFrame, static: np.ndarray) -> pd.DataFrame:
    conf = df["confidence"].astype(str).str.lower().str[0]
    df = df[conf != "l"]                                   # VIIRS confidence: l(ow), n(ominal), h(igh)
    if len(static):
        df = df[~np.isin(cell_key(df["latitude"], df["longitude"]), static)]
    return df


def run(settings: Settings) -> dict:
    out = settings.data_dir / "clean"
    files = fire_files(settings)
    if not files:
        raise RuntimeError("No FIRMS files found. Run:  backfill --source firms")

    log.info("Fires pass 1 of 2: finding permanent hotspots in %d files", len(files))
    static = static_cells(read_fires(p) for p in files)
    write_frame(static, out / "fire_static_cells.parquet")
    log.info("  %d permanent hotspot cells (flares, industry) will be ignored", len(static))
    static_ids = np.sort(static["cell"].to_numpy(dtype=np.int64))

    log.info("Fires pass 2 of 2: fire power around each city")
    parts, used = [], 0
    for region in sorted({c.fire_region for c in settings.cities}):
        cities = [c for c in settings.cities if c.fire_region == region]
        for i, path in enumerate(fire_files(settings, region), 1):
            df = keep_real_fires(read_fires(path), static_ids)
            used += len(df)
            for c in cities:
                rings = city_rings(df, c.slug, c.lat, c.lon)
                if len(rings):
                    parts.append(rings)
            if i % 500 == 0:
                log.info("  %s: %d files read", region, i)
        log.info("  %-15s done (%s)", region, ", ".join(c.slug for c in cities))
    cols = ["city", "period_utc", "ring_km", "sector", "frp", "n"]
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    # Both satellites and any split files land in the same period: add them up.
    table = table.groupby(cols[:4], as_index=False)[["frp", "n"]].sum().sort_values(cols[:4])
    write_frame(table.reset_index(drop=True), out / "fires_by_city.parquet")
    return {"files": len(files), "static_cells": int(len(static)), "detections_used": int(used),
            "rows": int(len(table))}
