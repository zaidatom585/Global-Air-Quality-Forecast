"""NASA FIRMS: satellite fire detections (VIIRS), searched by fire region box.

Files: data/firms/<region>/<satellite>/<window start>_<source>_<part start>.parquet. Windows are 10 days on a
fixed grid; a window that spans the end of the archive is saved in two parts (SP, then NRT).

FIRMS has two versions of each satellite's data:
- standard processing (<SAT>_SP): the archive, final and consistent, but a few months behind;
- near real time (<SAT>_NRT): the recent weeks, replaced by SP later.
The data availability endpoint says where SP ends; windows after that use NRT. When SP later covers a
window first saved from NRT, the NRT part is replaced.
"""
from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import pandas as pd

from aqcast.settings import HISTORY_START, Settings
from aqcast.storage import as_utc_midnight, fetched_after, utc_now, write_frame
from aqcast.web import Fetcher, MissingKey, RateLimiter, SourceError

log = logging.getLogger(__name__)

API = "https://firms.modaps.eosdis.nasa.gov/api"
WINDOW_DAYS = 10            # the area API returns at most 10 days per request
SETTLE_DAYS = 7             # late detections are added to NRT for a few days after the fact
SATELLITES = ("VIIRS_SNPP", "VIIRS_NOAA20")
KEEP = ["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "daynight", "bright_ti4", "bright_ti5",
        "scan", "track", "satellite", "instrument", "version"]


def fetcher(settings: Settings) -> Fetcher:
    if not settings.firms_key:
        raise MissingKey("FIRMS_MAP_KEY is not set")
    # FIRMS allows a few thousand transactions per 10 minutes; one request every 1.5 s stays far below that.
    return Fetcher("firms", RateLimiter(gap=1.5))


def availability(f: Fetcher, key: str) -> dict[str, tuple[date, date]]:
    """{source id: (first date, last date)} for every FIRMS source."""
    r = f.get(f"{API}/data_availability/csv/{key}/all")
    df = pd.read_csv(io.StringIO(r.text))
    if "data_id" not in df.columns:
        raise SourceError(f"FIRMS availability check failed: {r.text[:200]}")
    return {row.data_id: (date.fromisoformat(str(row.min_date)[:10]), date.fromisoformat(str(row.max_date)[:10]))
            for row in df.itertuples()}


def grid(start: date, end: date, anchor: date = HISTORY_START) -> list[date]:
    """Window starts every 10 days from a fixed anchor, so the same window always has the same file name."""
    first = anchor + timedelta(days=((start - anchor).days // WINDOW_DAYS) * WINDOW_DAYS)
    out, d = [], first
    while d <= end:
        out.append(d)
        d += timedelta(days=WINDOW_DAYS)
    return out


def plan_parts(w_start: date, satellite: str, avail: dict) -> list[tuple[date, int, str]]:
    """(part start, days, source) for one whole grid window: SP up to where the archive ends, NRT after it.

    Always the whole window (cut only by what FIRMS has), never a slice of it, so a short update can't
    replace a full window's file with fewer days.
    """
    lo, hi = max(w_start, HISTORY_START), w_start + timedelta(days=WINDOW_DAYS - 1)
    sp, nrt = avail.get(f"{satellite}_SP"), avail.get(f"{satellite}_NRT")
    parts = []
    if sp and lo <= sp[1] and hi >= sp[0]:
        a, b = max(lo, sp[0]), min(hi, sp[1])
        parts.append((a, (b - a).days + 1, f"{satellite}_SP"))
        lo = b + timedelta(days=1)
    if nrt and lo <= hi and hi >= nrt[0] and lo <= nrt[1]:
        a, b = max(lo, nrt[0]), min(hi, nrt[1])
        parts.append((a, (b - a).days + 1, f"{satellite}_NRT"))
    return parts


def parse(text: str) -> pd.DataFrame:
    head = text.lstrip()[:200].lower()
    if not head.startswith("latitude"):
        raise SourceError(f"FIRMS returned no CSV: {text[:200]}")
    df = pd.read_csv(io.StringIO(text))
    for col in KEEP:
        if col not in df:
            df[col] = pd.NA
    df = df[KEEP].copy()
    t = df["acq_time"].astype("Int64").astype(str).str.zfill(4)
    df["detected_utc"] = pd.to_datetime(df["acq_date"].astype(str) + " " + t, format="%Y-%m-%d %H%M", utc=True)
    df["confidence"] = df["confidence"].astype(str)
    df["version"] = df["version"].astype(str)
    return df


def fetch_window(f: Fetcher, key: str, source: str, box: tuple, start: date, days: int) -> pd.DataFrame:
    w, s, e, n = box
    url = f"{API}/area/csv/{key}/{source}/{w:g},{s:g},{e:g},{n:g}/{days}/{start:%Y-%m-%d}"
    df = parse(f.get(url).text)
    # Keep only detections inside the requested days (FIRMS dates are UTC).
    keep = (df["detected_utc"] >= pd.Timestamp(as_utc_midnight(start))) & \
           (df["detected_utc"] < pd.Timestamp(as_utc_midnight(start + timedelta(days=days))))
    return df[keep].reset_index(drop=True)


def window_dir(settings: Settings, region: str, satellite: str):
    return settings.data_dir / "firms" / region / satellite


def download(settings: Settings, regions: list[str], start: date, end: date, satellites=SATELLITES,
             f: Fetcher | None = None) -> dict:
    f = f or fetcher(settings)
    avail = availability(f, settings.firms_key)
    fetched = skipped = detections = 0
    for satellite in satellites:
        for region in regions:
            box = settings.fire_regions[region]
            folder = window_dir(settings, region, satellite)
            for w_start in grid(start, end):
                parts = plan_parts(w_start, satellite, avail)
                if not parts:
                    continue
                names = [folder / f"{w_start:%Y-%m-%d}_{src}_{p:%Y-%m-%d}.parquet" for p, _, src in parts]
                w_end = w_start + timedelta(days=WINDOW_DAYS)
                # A whole window from one source, saved a week after it ended, is final; when the archive later
                # covers a window saved from NRT, the SP file name doesn't exist yet, so it is fetched then.
                whole = len(parts) == 1 and parts[0][1] == WINDOW_DAYS
                if whole and fetched_after(names[0], as_utc_midnight(w_end) + timedelta(days=SETTLE_DAYS)):
                    skipped += 1
                    continue
                for (p_start, days, source), path in zip(parts, names):
                    df = fetch_window(f, settings.firms_key, source, box, p_start, days)
                    df["region"] = region
                    df["source"] = source
                    df["fetched_utc"] = utc_now()
                    write_frame(df, path)
                    fetched += 1
                    detections += len(df)
                for old in folder.glob(f"{w_start:%Y-%m-%d}_*.parquet"):
                    if old not in names:  # e.g. an NRT part now covered by the archive
                        old.unlink()
            log.info("FIRMS %-15s %s: %d requests so far, %d detections", region, satellite, fetched, detections)
    return {"parts_fetched": fetched, "windows_skipped": skipped, "detections": detections,
            "requests": f.requests_made}
