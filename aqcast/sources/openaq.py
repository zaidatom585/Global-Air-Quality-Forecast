"""OpenAQ: ground PM2.5 sensors.

Two steps:
1. choose_sensors: for each city, keep the best active PM2.5 sensors within 25 km
   (official monitors first, then the longest records). Saved to data/openaq/sensors.parquet.
2. download_hours: hourly averages per sensor, one file per sensor and month:
   data/openaq/hours/sensor=<id>/<YYYY-MM>.parquet
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from math import asin, cos, radians, sin, sqrt

import pandas as pd

from aqcast.settings import City, Settings
from aqcast.storage import as_utc_midnight, fetched_after, month_end, month_starts, read_frame, utc_now, write_frame
from aqcast.web import Fetcher, MissingKey, RateLimiter, SourceError

log = logging.getLogger(__name__)

API = "https://api.openaq.org/v3"
PM25 = 2                    # OpenAQ parameter id for PM2.5
RADIUS_M = 25_000           # the API's maximum search radius
SENSORS_PER_CITY = 3          # median of 3 sensors per city; more makes the backfill much longer
PAGE = 1000
LATE_DATA_DAYS = 3          # a month counts as complete once it was fetched this long after it ended


def fetcher(settings: Settings) -> Fetcher:
    if not settings.openaq_key:
        raise MissingKey("OPENAQ_API_KEY is not set")
    # Free tier: 60 requests a minute and 2,000 an hour. Stay a little under both.
    return Fetcher("openaq", RateLimiter(per_minute=55, per_hour=1900), headers={"X-API-Key": settings.openaq_key})


def _utc(value: dict | None) -> pd.Timestamp | None:
    v = (value or {}).get("utc")
    return pd.Timestamp(v) if v else None


def _km(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = radians(lat1), radians(lat2), radians(lon2 - lon1)
    a = sin((p2 - p1) / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


# ----------------------------------------------------------------------------------------------- sensors
def candidate_sensors(f: Fetcher, city: City) -> pd.DataFrame:
    r = f.get(f"{API}/locations", params={"coordinates": f"{city.lat},{city.lon}", "radius": RADIUS_M,
                                          "parameters_id": PM25, "limit": 1000})
    rows = []
    for loc in r.json().get("results", []):
        coords = loc.get("coordinates") or {}
        for s in loc.get("sensors", []):
            if (s.get("parameter") or {}).get("id") != PM25:
                continue
            rows.append({
                "city": city.slug,
                "sensor_id": int(s["id"]),
                "location_id": loc.get("id"),
                "location": loc.get("name"),
                "provider": (loc.get("provider") or {}).get("name"),
                "reference_monitor": bool(loc.get("isMonitor")),
                "lat": coords.get("latitude"),
                "lon": coords.get("longitude"),
                "first_utc": _utc(loc.get("datetimeFirst")),
                "last_utc": _utc(loc.get("datetimeLast")),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["distance_km"] = [round(_km(city.lat, city.lon, a, b), 1) if a is not None and b is not None else None
                             for a, b in zip(df["lat"], df["lon"])]
    return df


def rank_sensors(df: pd.DataFrame, now: datetime, keep: int = SENSORS_PER_CITY) -> pd.DataFrame:
    """Active in the last 7 days; official monitors first, then the longest record, then the closest."""
    if df.empty:
        return df
    active = df[pd.to_datetime(df["last_utc"], utc=True) >= pd.Timestamp(now) - pd.Timedelta(days=7)].copy()
    # The same physical sensor can appear under two location records; keep one per location name.
    active = active.drop_duplicates(subset=["location", "provider"], keep="first")
    active["first_sort"] = pd.to_datetime(active["first_utc"], utc=True).fillna(pd.Timestamp(now))
    active = active.sort_values(["reference_monitor", "first_sort", "distance_km"], ascending=[False, True, True])
    return active.drop(columns="first_sort").head(keep).reset_index(drop=True)


def choose_sensors(settings: Settings, cities: list[City], f: Fetcher | None = None) -> pd.DataFrame:
    f = f or fetcher(settings)
    now = utc_now()
    picked = []
    for city in cities:
        ranked = rank_sensors(candidate_sensors(f, city), now)
        log.info("%-14s %d sensors kept (%d official monitors)", city.name, len(ranked),
                 int(ranked["reference_monitor"].sum()) if not ranked.empty else 0)
        picked.append(ranked)
    out = pd.concat([p for p in picked if not p.empty], ignore_index=True)
    out["chosen_utc"] = now
    path = settings.data_dir / "openaq" / "sensors.parquet"
    if path.exists():  # keep sensors already chosen for cities not in this run
        old = read_frame(path)
        out = pd.concat([old[~old["city"].isin([c.slug for c in cities])], out], ignore_index=True)
    write_frame(out, path)
    return out


def load_sensors(settings: Settings, cities: list[City]) -> pd.DataFrame:
    path = settings.data_dir / "openaq" / "sensors.parquet"
    if not path.exists():
        raise SourceError("No sensors chosen yet. Run:  sensors")
    df = read_frame(path)
    return df[df["city"].isin([c.slug for c in cities])]


# ----------------------------------------------------------------------------------------------- hours
def fetch_hours(f: Fetcher, sensor_id: int, start: datetime, end: datetime) -> pd.DataFrame:
    """Hourly PM2.5 averages for one sensor with hour starts in [start, end)."""
    rows, page = [], 1
    while True:
        r = f.get(f"{API}/sensors/{sensor_id}/hours", params={
            "datetime_from": start.isoformat(), "datetime_to": end.isoformat(), "limit": PAGE, "page": page})
        results = r.json().get("results", [])
        for x in results:
            hour = _utc((x.get("period") or {}).get("datetimeFrom"))
            if hour is None or not (pd.Timestamp(start) <= hour < pd.Timestamp(end)):
                continue  # the date filter is also applied here, in case the API ignores it
            rows.append({"sensor_id": sensor_id, "hour_utc": hour, "pm25": x.get("value"),
                         "pct_complete": (x.get("coverage") or {}).get("percentComplete")})
        if len(results) < PAGE or page >= 50:
            break
        page += 1
    df = pd.DataFrame(rows, columns=["sensor_id", "hour_utc", "pm25", "pct_complete"])
    return df.drop_duplicates(subset=["sensor_id", "hour_utc"], keep="last").sort_values("hour_utc")


def month_path(settings: Settings, sensor_id: int, month: date):
    return settings.data_dir / "openaq" / "hours" / f"sensor={sensor_id}" / f"{month:%Y-%m}.parquet"


def download_hours(settings: Settings, cities: list[City], start: date, end: date, f: Fetcher | None = None) -> dict:
    """Backfill: one request (or a few pages) per sensor and month; complete months already on disk are skipped."""
    f = f or fetcher(settings)
    sensors = load_sensors(settings, cities)
    done = skipped = rows = 0
    for s in sensors.itertuples():
        first = pd.Timestamp(s.first_utc).date() if pd.notna(s.first_utc) else start
        for m in month_starts(max(start, date(first.year, first.month, 1)), end):
            path = month_path(settings, s.sensor_id, m)
            complete_after = as_utc_midnight(month_end(m)) + timedelta(days=LATE_DATA_DAYS)
            if fetched_after(path, complete_after):
                skipped += 1
                continue
            lo = as_utc_midnight(m)                       # the whole month, never a slice of it
            hi = as_utc_midnight(month_end(m)) + timedelta(days=1)
            df = fetch_hours(f, s.sensor_id, lo, hi)
            df["fetched_utc"] = utc_now()
            write_frame(df, path)
            done += 1
            rows += len(df)
        log.info("%-14s sensor %-9s %s: months fetched so far %d", s.city, s.sensor_id, s.location, done)
    return {"months_fetched": done, "months_skipped": skipped, "rows": rows, "requests": f.requests_made}


def update_recent(settings: Settings, cities: list[City], hours: int = 72, f: Fetcher | None = None) -> dict:
    """Hourly update: fetch the last `hours` for every sensor and merge them into the month files."""
    f = f or fetcher(settings)
    sensors = load_sensors(settings, cities)
    end = utc_now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    start = end - timedelta(hours=hours)
    rows = 0
    for s in sensors.itertuples():
        new = fetch_hours(f, s.sensor_id, start, end)
        if new.empty:
            continue
        # Rows from updates carry no fetch time: only a backfill after the month ended marks a month complete,
        # so a month pieced together from hourly updates (with possible gaps) is re-downloaded whole later.
        new["fetched_utc"] = pd.Series(pd.NaT, index=new.index, dtype="datetime64[ns, UTC]")
        for m, part in new.groupby(new["hour_utc"].dt.strftime("%Y-%m")):
            path = month_path(settings, s.sensor_id, date.fromisoformat(m + "-01"))
            if path.exists():
                old = read_frame(path)
                old = old[~old["hour_utc"].isin(part["hour_utc"])]
                merged = pd.concat([old, part], ignore_index=True)
            else:
                merged = part
            write_frame(merged.sort_values("hour_utc"), path)
            rows += len(part)
    return {"rows": rows, "requests": f.requests_made, "window": f"{start:%Y-%m-%d %H:00} to {end:%Y-%m-%d %H:00} UTC"}
