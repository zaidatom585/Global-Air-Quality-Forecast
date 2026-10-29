"""OpenAQ: ground PM2.5 sensors.

Two steps:
1. choose_sensors: for each city, keep the 3 best active PM2.5 stations ("sites") within 25 km (official
   monitors first, then the longest records) with ALL their sensor records: OpenAQ lists many stations
   several times, under old and re-registered records and old and new sensor ids, and only together do they
   hold the full history. Each sensor's own first and last dates are looked up. Saved to
   data/openaq/sensors.parquet.
2. download_hours: hourly averages per sensor over its own active period, one file per sensor and month:
   data/openaq/hours/sensor=<id>/<YYYY-MM>.parquet
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from math import asin, cos, radians, sin, sqrt

import pandas as pd

from aqcast.settings import HISTORY_START, City, Settings
from aqcast.storage import as_utc_midnight, fetched_after, month_end, month_starts, read_frame, utc_now, write_frame
from aqcast.web import Fetcher, MissingKey, RateLimiter, SourceError

log = logging.getLogger(__name__)

API = "https://api.openaq.org/v3"
PM25 = 2                    # OpenAQ parameter id for PM2.5
RADIUS_M = 25_000           # the API's maximum search radius
SITES_PER_CITY = 3          # stations per city; each may have several sensor records
PAGE = 1000
LATE_DATA_DAYS = 3          # a month counts as complete once it was fetched this long after it ended


def fetcher(settings: Settings) -> Fetcher:
    if not settings.openaq_key:
        raise MissingKey("OPENAQ_API_KEY is not set")
    # Free tier: 60 requests a minute and 2,000 an hour. Stay a little under both.
    return Fetcher("openaq", RateLimiter(per_minute=55, per_hour=1900), headers={"X-API-Key": settings.openaq_key},
                   retries=4)


def _utc(value: dict | None) -> pd.Timestamp | None:
    v = (value or {}).get("utc")
    return pd.Timestamp(v) if v else None


def _km(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = radians(lat1), radians(lat2), radians(lon2 - lon1)
    a = sin((p2 - p1) / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


# ----------------------------------------------------------------------------------------------- sensors
def find_locations(f: Fetcher, city: City) -> list[dict]:
    """PM2.5 locations around a city. OpenAQ's search fails for some places (HTTP 500 every time), so if the
    usual 25 km radius search fails, try a box of the same size, then a smaller radius."""
    dlat = RADIUS_M / 111_000
    dlon = dlat / max(cos(radians(city.lat)), 0.2)
    attempts = [
        ("25 km radius", {"coordinates": f"{city.lat},{city.lon}", "radius": RADIUS_M}),
        ("box", {"bbox": f"{city.lon - dlon:.4f},{city.lat - dlat:.4f},{city.lon + dlon:.4f},{city.lat + dlat:.4f}"}),
        ("10 km radius", {"coordinates": f"{city.lat},{city.lon}", "radius": 10_000}),
    ]
    errors: list[str] = []
    last: Exception | None = None
    for how, params in attempts:
        try:
            r = f.get(f"{API}/locations", params={**params, "parameters_id": PM25, "limit": 1000}, retries=2)
            if errors:
                log.info("%-14s found with a %s search after: %s", city.name, how, "; ".join(errors))
            return r.json().get("results", [])
        except SourceError as e:
            errors.append(f"{how} failed")
            last = e
    raise SourceError(f"every search failed ({'; '.join(errors)}): {last}")


def candidate_sensors(f: Fetcher, city: City) -> pd.DataFrame:
    rows = []
    for loc in find_locations(f, city):
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


SITE_RADIUS_KM = 0.5         # location records this close together are one physical station
# Test, demo and calibration units aren't monitoring stations.
NOT_A_STATION = re.compile(r"\b(?:test|demo|testing|trial|prototype|calib\.?|calibration)(?:\b|\s)", re.I)


def group_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Give every location record a site id: records within SITE_RADIUS_KM of each other are one station.

    OpenAQ often lists one station several times (old and re-registered records, and old and new sensor ids
    on one record); together they hold the station's full history.
    """
    df = df.copy()
    locs = df.drop_duplicates("location_id")[["location_id", "lat", "lon"]].sort_values("location_id")
    site_of: dict = {}
    centres: list[tuple] = []                     # (site id, lat, lon)
    for loc in locs.itertuples():
        site = None
        if loc.lat is not None and loc.lon is not None and pd.notna(loc.lat):
            for sid, la, lo in centres:
                if _km(loc.lat, loc.lon, la, lo) <= SITE_RADIUS_KM:
                    site = sid
                    break
        if site is None:
            site = loc.location_id
            centres.append((site, loc.lat, loc.lon))
        site_of[loc.location_id] = site
    df["site_id"] = df["location_id"].map(site_of)
    return df


def rank_sites(df: pd.DataFrame, now: datetime, keep: int = SITES_PER_CITY) -> pd.DataFrame:
    """Every PM2.5 sensor of the best `keep` active sites: official monitors first, then the longest record,
    then the closest. A site is active if any of its records reported in the last 7 days."""
    if df.empty:
        return df
    df = df[~df["location"].fillna("").str.contains(NOT_A_STATION)]
    if df.empty:
        return df
    df = group_sites(df)
    df["first_ts"] = pd.to_datetime(df["first_utc"], utc=True)
    df["last_ts"] = pd.to_datetime(df["last_utc"], utc=True)
    sites = df.groupby("site_id").agg(monitor=("reference_monitor", "max"), first=("first_ts", "min"),
                                      last=("last_ts", "max"), distance=("distance_km", "min"))
    cutoff = pd.Timestamp(now - timedelta(days=7))
    sites = sites[sites["last"] >= cutoff]
    sites["first"] = sites["first"].fillna(pd.Timestamp(now))
    best = sites.sort_values(["monitor", "first", "distance"], ascending=[False, True, True]).head(keep).index
    out = df[df["site_id"].isin(best)].copy()
    # The site's name is that of its most recent record.
    names = out.sort_values("last_ts").groupby("site_id")["location"].last()
    out["site"] = out["site_id"].map(names)
    out["site_rank"] = out["site_id"].map({sid: i + 1 for i, sid in enumerate(best)})
    out = out.drop(columns=["first_ts", "last_ts"]).drop_duplicates("sensor_id")
    return out.sort_values(["site_rank", "sensor_id"]).reset_index(drop=True)


def sensor_dates(f: Fetcher, sensor_id: int) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """First and last reading of one sensor (more precise than its location record's dates)."""
    try:
        res = f.get(f"{API}/sensors/{sensor_id}").json().get("results") or [{}]
    except SourceError as e:
        log.warning("sensor %s: could not read its dates (%s)", sensor_id, e)
        return None, None
    return _utc(res[0].get("datetimeFirst")), _utc(res[0].get("datetimeLast"))


def choose_sensors(settings: Settings, cities: list[City], f: Fetcher | None = None) -> pd.DataFrame:
    f = f or fetcher(settings)
    now = utc_now()
    start = pd.Timestamp(as_utc_midnight(HISTORY_START))
    picked, failed = [], []
    for city in cities:
        try:
            ranked = rank_sites(candidate_sensors(f, city), now)
        except SourceError as e:   # one city's OpenAQ error shouldn't stop the others; rerun with --cities later
            log.warning("%-14s skipped: %s", city.name, e)
            failed.append(city.slug)
            continue
        if ranked.empty:
            log.warning("%-14s no active PM2.5 site found", city.name)
            continue
        dates = [sensor_dates(f, sid) for sid in ranked["sensor_id"]]
        # Prefer each sensor's own dates; fall back to its location record's.
        ranked["first_utc"] = [d[0] if d[0] is not None else r for d, r in zip(dates, ranked["first_utc"])]
        ranked["last_utc"] = [d[1] if d[1] is not None else r for d, r in zip(dates, ranked["last_utc"])]
        last = pd.to_datetime(ranked["last_utc"], utc=True)
        ranked = ranked[last.isna() | (last >= start)]              # sensors that ended before Aug 2022 add nothing
        log.info("%-14s %d sites, %d sensors (%s)", city.name, ranked["site_id"].nunique(), len(ranked),
                 "; ".join(f"{n}: {k}" for n, k in ranked.groupby("site", sort=False)["sensor_id"].count().items()))
        picked.append(ranked)
    if not picked:
        raise SourceError(f"no city could be checked (failed: {', '.join(failed)})")
    out = pd.concat(picked, ignore_index=True)
    out["chosen_utc"] = now
    path = settings.data_dir / "openaq" / "sensors.parquet"
    if path.exists():  # keep sensors already chosen for cities not redone in this run (including failed ones)
        old = read_frame(path)
        if "site_id" in old:
            out = pd.concat([old[~old["city"].isin(out["city"].unique())], out], ignore_index=True)
    write_frame(out, path)
    if failed:
        log.warning("Cities skipped because of OpenAQ errors: %s. Rerun:  sensors --cities %s",
                    ", ".join(failed), ",".join(failed))
    return out


def load_sensors(settings: Settings, cities: list[City]) -> pd.DataFrame:
    path = settings.data_dir / "openaq" / "sensors.parquet"
    if not path.exists():
        raise SourceError("No sensors chosen yet. Run:  sensors")
    df = read_frame(path)
    if "site_id" not in df:
        raise SourceError("sensors.parquet is from an older version. Run:  sensors")
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
    """Backfill: one request (or a few pages) per sensor and month, over each sensor's own active period;
    complete months already on disk are skipped, and a sensor that keeps failing is skipped, not fatal."""
    f = f or fetcher(settings)
    sensors = load_sensors(settings, cities)
    done = skipped = rows = 0
    failed = []
    for s in sensors.itertuples():
        first = pd.Timestamp(s.first_utc).date() if pd.notna(s.first_utc) else start
        last = pd.Timestamp(s.last_utc).date() if pd.notna(s.last_utc) else end
        months = month_starts(max(start, date(first.year, first.month, 1)), min(end, last))
        fetched_here = 0
        try:
            for m in months:
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
                fetched_here += 1
                rows += len(df)
        except SourceError as e:
            log.warning("%s sensor %s: stopped after repeated errors (%s); rerun to resume", s.city, s.sensor_id, e)
            # fall through: the log line below reports how far this sensor got
            failed.append(int(s.sensor_id))
        log.info("%-14s sensor %-9s %-36.36s %s to %s: %d of %d months fetched", s.city, s.sensor_id, s.location,
                 months[0] if months else "-", months[-1] if months else "-", fetched_here, len(months))
    return {"months_fetched": done, "months_skipped": skipped, "rows": rows, "failed_sensors": failed,
            "requests": f.requests_made}


def update_recent(settings: Settings, cities: list[City], hours: int = 72, f: Fetcher | None = None) -> dict:
    """Hourly update: fetch the last `hours` for every sensor and merge them into the month files."""
    f = f or fetcher(settings)
    sensors = load_sensors(settings, cities)
    end = utc_now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    start = end - timedelta(hours=hours)
    # Only sensors still reporting when the list was made; retired records have nothing new.
    chosen = pd.to_datetime(sensors["chosen_utc"], utc=True)
    sensors = sensors[pd.to_datetime(sensors["last_utc"], utc=True) >= chosen - pd.Timedelta(days=14)]
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
