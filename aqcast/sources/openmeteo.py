"""Open-Meteo: weather and the CAMS air-quality forecast, per city.

History (for training), one file per city and month:
  data/weather/history/city=<slug>/<YYYY-MM>.parquet    hourly weather (reanalysis)
  data/cams/history/city=<slug>/<YYYY-MM>.parquet       hourly CAMS PM2.5, PM10, dust, aerosols, CO

Forecast snapshots (what was actually forecast, saved when it was issued), every 6 hours:
  data/forecasts/weather/issued=<YYYY-MM-DDTHH>/<slug>.parquet
  data/forecasts/cams/issued=<YYYY-MM-DDTHH>/<slug>.parquet

Why snapshots: Open-Meteo's CAMS history is stitched from the first hours of each forecast run, so it shows
roughly what CAMS knew at the time, not what it forecast 24-72 hours earlier. A fair "model vs CAMS forecast"
comparison needs the real forecasts, so they are saved from the first day of collection onwards.

Request budget: Open-Meteo's free tier allows 600 calls a minute, 5,000 an hour and 10,000 a day, and counts
a request covering more than 2 weeks or more than 10 variables as several calls. Each request here passes that
weight to the rate limiter, and the hourly update only refreshes what is stale, so collection stays well
inside the daily limit.

Free use is non-commercial and needs attribution: weather by Open-Meteo, air quality by CAMS (Copernicus).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from math import ceil

import pandas as pd

from aqcast.settings import HISTORY_START, City, Settings
from aqcast.storage import as_utc_midnight, fetched_after, month_end, month_starts, utc_now, write_frame
from aqcast.web import Fetcher, RateLimiter, SharedLimiter, SourceError

log = logging.getLogger(__name__)

WEATHER_HISTORY = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure",
                "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
CAMS_VARS = ["pm2_5", "pm10", "dust", "aerosol_optical_depth", "carbon_monoxide"]
REANALYSIS_LAG_DAYS = 6       # the weather archive trails real time by about 5 days
FORECAST_DAYS_WEATHER = 4     # covers the 72-hour horizon plus a margin
FORECAST_DAYS_CAMS = 5
SNAPSHOT_EVERY_HOURS = 6      # weather models and CAMS issue new runs every 6-12 hours
REFRESH_HOURS = {"weather": 24, "cams": 6}   # how often the hourly update re-downloads the current month


def fetcher(settings: Settings | None = None) -> Fetcher:
    """Stay under the free tier's 600 a minute, 5,000 an hour and 10,000 a day, counted by weight. The count is
    kept in a file under data/, so backfills running in parallel windows share one budget."""
    if settings is None:
        return Fetcher("open-meteo", RateLimiter(per_minute=400, per_hour=4000, per_day=9000, gap=0.2))
    ledger = settings.data_dir / "state" / "open-meteo-calls.log"
    return Fetcher("open-meteo", SharedLimiter(ledger, per_minute=400, per_hour=4000, per_day=9000, gap=0.2))


def weight(days: int, n_vars: int) -> float:
    """How many calls Open-Meteo counts for one request: per 2 weeks of data and per 10 variables."""
    return max(1.0, ceil(days / 14)) * max(1.0, ceil(n_vars / 10))


def hourly_frame(payload: dict, variables: list[str]) -> pd.DataFrame:
    if payload.get("error"):
        raise SourceError(f"Open-Meteo: {payload.get('reason')}")
    hourly = payload.get("hourly") or {}
    if "time" not in hourly:
        raise SourceError(f"Open-Meteo: no hourly data in response (keys: {sorted(payload)})")
    df = pd.DataFrame({"time_utc": pd.to_datetime(hourly["time"], utc=True)})
    for v in variables:
        df[v] = pd.to_numeric(pd.Series(hourly.get(v, [None] * len(df))), errors="coerce")
    return df


def _params(city: City, variables: list[str], **extra) -> dict:
    return {"latitude": city.lat, "longitude": city.lon, "hourly": ",".join(variables), "timezone": "GMT",
            "wind_speed_unit": "ms", **extra}


# ------------------------------------------------------------------------------------------------ history
KINDS = {
    "weather": {"url": WEATHER_HISTORY, "vars": WEATHER_VARS, "extra": {}, "settle_days": REANALYSIS_LAG_DAYS + 4},
    "cams": {"url": AIR_QUALITY, "vars": CAMS_VARS, "extra": {"domains": "cams_global"}, "settle_days": 2},
}


def history_path(settings: Settings, kind: str, city: City, month: date):
    return settings.data_dir / kind / "history" / f"city={city.slug}" / f"{month:%Y-%m}.parquet"


def _fresh(path, hours: float) -> bool:
    """True if the file was downloaded less than `hours` ago."""
    return fetched_after(path, utc_now() - timedelta(hours=hours))


def download_history(settings: Settings, cities: list[City], start: date, end: date, kind: str,
                     refresh_hours: float | None = None, f: Fetcher | None = None) -> dict:
    """One file per city and month. A month already downloaded after it ended is skipped. With refresh_hours,
    an unfinished month is only re-downloaded if its file is older than that (used by the hourly update)."""
    f = f or fetcher(settings)
    spec = KINDS[kind]
    last_day = date.today() - timedelta(days=REANALYSIS_LAG_DAYS) if kind == "weather" else date.today()
    end = min(end, last_day)
    fetched = skipped = rows = 0
    for city in cities:
        for m in month_starts(max(start, HISTORY_START), end):
            path = history_path(settings, kind, city, m)
            if fetched_after(path, as_utc_midnight(month_end(m)) + timedelta(days=spec["settle_days"])):
                skipped += 1
                continue
            if refresh_hours is not None and _fresh(path, refresh_hours):
                skipped += 1
                continue
            a, b = max(m, HISTORY_START), min(month_end(m), last_day)   # the whole month, never a slice of it
            r = f.get(spec["url"], params=_params(city, spec["vars"], start_date=a.isoformat(), end_date=b.isoformat(),
                                                  **spec["extra"]),
                      weight=weight((b - a).days + 1, len(spec["vars"])))
            df = hourly_frame(r.json(), spec["vars"])
            df.insert(0, "city", city.slug)
            df["fetched_utc"] = utc_now()
            write_frame(df, path)
            fetched += 1
            rows += len(df)
        log.info("%s history %-14s done (%d files fetched so far)", kind, city.name, fetched)
    return {"files_fetched": fetched, "files_skipped": skipped, "rows": rows, "requests": f.requests_made}


def download_weather(settings: Settings, cities: list[City], start: date, end: date, **kw) -> dict:
    return download_history(settings, cities, start, end, "weather", **kw)


def download_cams(settings: Settings, cities: list[City], start: date, end: date, **kw) -> dict:
    return download_history(settings, cities, start, end, "cams", **kw)


# ---------------------------------------------------------------------------------------------- snapshots
def issue_time():
    """The current 6-hour slot (00, 06, 12 or 18 UTC): one snapshot per slot however often update runs."""
    now = utc_now()
    return now.replace(hour=now.hour - now.hour % SNAPSHOT_EVERY_HOURS, minute=0, second=0, microsecond=0)


def snapshot(settings: Settings, cities: list[City], f: Fetcher | None = None) -> dict:
    """Save the current weather and CAMS forecasts for every city, once per 6-hour slot."""
    f = f or fetcher(settings)
    issued = issue_time()
    stamp = f"issued={issued:%Y-%m-%dT%H}"
    saved = 0
    for city in cities:
        for kind, url, variables, days, extra in (
                ("weather", WEATHER_FORECAST, WEATHER_VARS, FORECAST_DAYS_WEATHER, {}),
                ("cams", AIR_QUALITY, CAMS_VARS, FORECAST_DAYS_CAMS, {"domains": "cams_global"})):
            path = settings.data_dir / "forecasts" / kind / stamp / f"{city.slug}.parquet"
            if path.exists():
                continue
            r = f.get(url, params=_params(city, variables, forecast_days=days, **extra),
                      weight=weight(days, len(variables)))
            df = hourly_frame(r.json(), variables)
            df = df[df["time_utc"] >= pd.Timestamp(issued)].reset_index(drop=True)
            df.insert(0, "city", city.slug)
            df.insert(1, "issued_utc", issued)
            df.insert(3, "lead_hours", ((df["time_utc"] - pd.Timestamp(issued)) / pd.Timedelta(hours=1)).astype(int))
            df["saved_utc"] = utc_now()
            write_frame(df, path)
            saved += 1
    return {"issued_utc": issued.isoformat(), "files_saved": saved, "requests": f.requests_made}

