"""The training table: one row per city, forecast time and horizon.

Forecast ("issue") times are every 6 hours (00, 06, 12, 18 UTC), like the CAMS and weather model runs. For each
issue time t and horizon h of 24, 48 or 72 hours, the target is the city's mean PM2.5 over the six hours that
end at t + h. Everything else in the row is known at t, except the weather at the target time (see below).

Feature groups (column prefixes):
  pm_     PM2.5 up to t: last 3 hours, last 24 and 72 hours, 24-hour maximum, the same time yesterday, trend
  cams_   CAMS at t: PM2.5, PM10, dust, aerosol optical depth, carbon monoxide; CAMS PM2.5 over the last day
  wx_     weather at t, and over the six target hours (wx_target_): temperature, humidity, wind, rain,
          pressure, cloud
  fire_   fire power in the last 24 and 72 hours per distance ring, all directions and upwind only
  time_   local hour and season at the target time, day of week, horizon

Weather at the target time comes from reanalysis, which is what happened, not what was forecast. In real use
the model gets the weather forecast instead, which is close for 1-3 days but not exact, so scores on this
table are a little optimistic. The saved forecast snapshots make the honest comparison possible later.
CAMS at the target time is left out on purpose: its history is stitched from analyses, so it would leak
what actually happened.

Output: data/features/training.parquet
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aqcast.fires import RINGS_KM, SECTORS
from aqcast.settings import City, Settings
from aqcast.storage import as_utc, read_frame, write_frame

log = logging.getLogger(__name__)

HORIZONS = (24, 48, 72)
ISSUE_EVERY = 6
TARGET_HOURS = 6                 # target = mean of the 6 hours ending at t + h
MIN_TARGET_HOURS = 3
WX = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure",
      "cloud_cover", "wind_speed_10m", "wind_direction_10m"]
CAMS = {"pm2_5": "pm25", "pm10": "pm10", "dust": "dust", "aerosol_optical_depth": "aod",
        "carbon_monoxide": "co"}


def hourly_index(start, end) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", tz="UTC")


def _history(settings: Settings, kind: str, city: City, columns: list[str]) -> pd.DataFrame:
    files = sorted((settings.data_dir / kind / "history" / f"city={city.slug}").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC", name="time_utc"), dtype=float)
    df = pd.concat([read_frame(p) for p in files], ignore_index=True)
    df["time_utc"] = as_utc(df["time_utc"])
    keep = ["time_utc", *[c for c in columns if c in df]]
    return df[keep].drop_duplicates("time_utc", keep="last").set_index("time_utc").sort_index()


def col(df: pd.DataFrame, name: str, index: pd.DatetimeIndex) -> pd.Series:
    """A float column, or all-missing if the source doesn't have it."""
    return pd.to_numeric(df[name], errors="coerce") if name in df else pd.Series(np.nan, index=index, dtype=float)


def wind_uv(speed: pd.Series, direction_from: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Wind as east (u) and north (v) parts of the direction it blows towards, so it can be averaged."""
    rad = np.radians(direction_from)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def upwind_weights(wind_from_deg: pd.Series) -> np.ndarray:
    """For each time, how much each compass sector lies upwind: cos of the angle between the sector's centre
    and the direction the wind comes from, 0 for sectors downwind. Shape (times, SECTORS)."""
    centres = np.arange(SECTORS) * (360 / SECTORS)
    diff = np.radians(np.asarray(wind_from_deg, dtype=float)[:, None] - centres[None, :])
    return np.clip(np.cos(diff), 0, None)


def fire_features(fires: pd.DataFrame, issue: pd.DatetimeIndex, wind_from: pd.Series) -> pd.DataFrame:
    """Fire power in the 24 and 72 hours before each issue time, per ring: total and upwind-weighted.
    A 6-hour period counts once it has ended, i.e. periods starting at or before t - 6 h."""
    out = pd.DataFrame(index=issue)
    rings = RINGS_KM[:-1]
    if fires.empty:
        for r in rings:
            for w in ("24h", "72h"):
                out[f"fire_frp_{w}_r{r}"] = 0.0
            out[f"fire_upwind_24h_r{r}"] = 0.0
        return out
    periods = pd.date_range(issue.min() - pd.Timedelta(hours=72), issue.max(), freq=f"{ISSUE_EVERY}h", tz="UTC")
    weights = upwind_weights(wind_from.reindex(issue).fillna(0.0).to_numpy())
    for r in rings:
        g = fires[fires["ring_km"] == r]
        grid = (g.pivot_table(index="period_utc", columns="sector", values="frp", aggfunc="sum")
                 .reindex(index=periods, columns=range(SECTORS)).fillna(0.0))
        sums = {w: grid.rolling(n, min_periods=1).sum() for w, n in (("24h", 4), ("72h", 12))}
        ended = issue - pd.Timedelta(hours=ISSUE_EVERY)               # the last period that has finished
        s24 = sums["24h"].reindex(ended).to_numpy()
        out[f"fire_frp_24h_r{r}"] = s24.sum(axis=1)
        out[f"fire_frp_72h_r{r}"] = sums["72h"].reindex(ended).to_numpy().sum(axis=1)
        out[f"fire_upwind_24h_r{r}"] = (s24 * weights).sum(axis=1)
    return out.fillna(0.0)


def city_table(city: City, pm: pd.Series, wx: pd.DataFrame, cams: pd.DataFrame, fires: pd.DataFrame) -> pd.DataFrame:
    """All rows for one city. `pm` is the hourly city PM2.5 indexed by hour (UTC)."""
    if pm.dropna().empty:
        return pd.DataFrame()
    idx = hourly_index(pm.index.min(), pm.index.max() + pd.Timedelta(hours=1))
    # OpenAQ stamps an hour by its start: the value for 06:00-07:00 is known only at 07:00. Shifting by one
    # hour puts each value at the time it becomes known, so features at t use only hours that have ended.
    p = pm.reindex(idx).shift(1)
    feats = pd.DataFrame(index=idx)
    feats["pm_3h"] = p.rolling(3, min_periods=1).mean()
    feats["pm_24h"] = p.rolling(24, min_periods=12).mean()
    feats["pm_72h"] = p.rolling(72, min_periods=36).mean()
    feats["pm_max_24h"] = p.rolling(24, min_periods=12).max()
    feats["pm_yesterday"] = feats["pm_3h"].shift(24)
    feats["pm_trend_24h"] = feats["pm_3h"] - feats["pm_yesterday"]

    w = wx.reindex(idx)
    u, v = wind_uv(col(w, "wind_speed_10m", idx), col(w, "wind_direction_10m", idx))
    now_cols = {"temperature_2m": "temp", "relative_humidity_2m": "rh", "surface_pressure": "pressure",
                "cloud_cover": "cloud", "wind_speed_10m": "wind"}
    for src, name in now_cols.items():
        feats[f"wx_{name}"] = col(w, src, idx)
    feats["wx_wind_u"], feats["wx_wind_v"] = u, v
    feats["wx_rain_24h"] = col(w, "precipitation", idx).rolling(24, min_periods=12).sum()

    c = cams.reindex(idx)
    for src, name in CAMS.items():
        feats[f"cams_{name}"] = col(c, src, idx)
    feats["cams_pm25_24h"] = col(c, "pm2_5", idx).rolling(24, min_periods=12).mean()

    issue = idx[(idx.hour % ISSUE_EVERY == 0)]
    base = feats.reindex(issue)
    base = base[base["pm_3h"].notna()]
    if base.empty:
        return pd.DataFrame()
    wind_from = col(w, "wind_direction_10m", idx)
    base = base.join(fire_features(fires, base.index, wind_from))

    # Targets and weather over the six hours ending at t + h (hours starting t+h-6 to t+h-1).
    target = p.rolling(TARGET_HOURS, min_periods=MIN_TARGET_HOURS).mean()
    blocks = {"temp": col(w, "temperature_2m", idx), "rh": col(w, "relative_humidity_2m", idx),
              "pressure": col(w, "surface_pressure", idx), "cloud": col(w, "cloud_cover", idx),
              "wind": col(w, "wind_speed_10m", idx), "wind_u": u, "wind_v": v}
    block_means = {k: s.rolling(TARGET_HOURS, min_periods=1).mean() for k, s in blocks.items()}
    rain_sum = col(w, "precipitation", idx).rolling(TARGET_HOURS, min_periods=1).sum()

    rows = []
    for h in HORIZONS:
        at = base.index + pd.Timedelta(hours=h)
        t = base.copy()
        t.insert(0, "horizon_h", h)
        t["target_utc"] = at
        t["y"] = target.reindex(at).to_numpy()
        for k, s in block_means.items():
            t[f"wx_target_{k}"] = s.reindex(at).to_numpy()
        t["wx_target_rain"] = rain_sum.reindex(at).to_numpy()
        local_hour = (at.hour + city.lon / 15.0) % 24
        t["time_local_hour_sin"] = np.sin(2 * np.pi * local_hour / 24)
        t["time_local_hour_cos"] = np.cos(2 * np.pi * local_hour / 24)
        doy = at.dayofyear.to_numpy()
        # Seasons are mirrored south of the equator, so shift the day of the year by half a year there.
        doy = (doy + (182 if city.lat < 0 else 0)) % 365
        t["time_season_sin"] = np.sin(2 * np.pi * doy / 365.25)
        t["time_season_cos"] = np.cos(2 * np.pi * doy / 365.25)
        t["time_weekday"] = at.dayofweek
        rows.append(t)
    out = pd.concat(rows)
    out = out[out["y"].notna()].copy()
    out.index.name = "issue_utc"
    out = out.reset_index()
    out.insert(0, "city", city.slug)
    out["lat"], out["lon"] = city.lat, city.lon
    return out


def run(settings: Settings) -> dict:
    d = settings.data_dir
    pm_all = read_frame(d / "clean" / "pm25_hourly.parquet")
    pm_all["hour_utc"] = as_utc(pm_all["hour_utc"])
    fire_path = d / "clean" / "fires_by_city.parquet"
    fires_all = read_frame(fire_path) if fire_path.exists() else pd.DataFrame(
        columns=["city", "period_utc", "ring_km", "sector", "frp", "n"])
    if fire_path.exists():
        fires_all["period_utc"] = as_utc(fires_all["period_utc"])
    else:
        log.warning("No fire table (run: fires); fire features will be 0")
    tables = []
    for city in settings.cities:
        pm = pm_all[pm_all["city"] == city.slug].set_index("hour_utc")["pm25"].sort_index()
        wx = _history(settings, "weather", city, WX)
        cams = _history(settings, "cams", city, list(CAMS))
        t = city_table(city, pm, wx, cams, fires_all[fires_all["city"] == city.slug])
        log.info("%-14s %7d rows", city.name, len(t))
        if len(t):
            tables.append(t)
    if not tables:
        raise RuntimeError("No training rows: run the clean step first")
    table = pd.concat(tables, ignore_index=True)
    write_frame(table, d / "features" / "training.parquet")
    missing = table.isna().mean().mul(100).round(1)
    worst = missing[missing > 5].sort_values(ascending=False)
    if len(worst):
        log.info("Columns missing in more than 5%% of rows: %s",
                 ", ".join(f"{k} {v}%" for k, v in worst.head(10).items()))
    return {"rows": int(len(table)), "columns": int(table.shape[1]), "cities": int(table["city"].nunique()),
            "first_issue": str(table["issue_utc"].min()), "last_issue": str(table["issue_utc"].max())}
