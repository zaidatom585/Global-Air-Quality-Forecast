"""Command line for collecting data. Runs the same way on a laptop, in Docker Compose and in Kubernetes.

    python -m aqcast.sync sensors                          choose OpenAQ sensors for each city (run first)
    python -m aqcast.sync backfill --source all            download history since Aug 2022 (resumable)
    python -m aqcast.sync backfill --source firms --start 2025-01-01 --cities delhi,lahore
    python -m aqcast.sync update                           recent hours + today's forecasts (run hourly)
    python -m aqcast.sync snapshot                         only save the current forecasts (once per 6 hours)
    python -m aqcast.sync status                           what is on disk

A source whose key is missing is skipped with a warning, so the others still run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from aqcast import settings as cfg
from aqcast.sources import firms, openaq, openmeteo
from aqcast.storage import utc_now
from aqcast.web import MissingKey, SourceError

log = logging.getLogger("aqcast")
SOURCES = ("openaq", "firms", "weather", "cams")


def _run(name: str, fn, results: dict) -> None:
    started = time.monotonic()
    try:
        results[name] = {"ok": True, **fn()}
    except MissingKey as e:
        log.warning("%s skipped: %s", name, e)
        results[name] = {"ok": False, "skipped": True, "error": str(e)}
    except SourceError as e:
        log.error("%s: %s", name, e)
        results[name] = {"ok": False, "error": str(e)}
    results[name]["seconds"] = round(time.monotonic() - started, 1)
    log.info("%s: %s", name, json.dumps(results[name], default=str))


def cmd_sensors(s: cfg.Settings, args) -> dict:
    results: dict = {}
    def choose():
        out = openaq.choose_sensors(s, s.pick(args.cities))
        return {"sensors": len(out), "cities": int(out["city"].nunique())}
    _run("openaq_sensors", choose, results)
    return results


def cmd_backfill(s: cfg.Settings, args) -> dict:
    cities = s.pick(args.cities)
    start = args.start or cfg.HISTORY_START
    end = args.end or date.today()
    regions = sorted({c.fire_region for c in cities})
    wanted = SOURCES if args.source == "all" else (args.source,)
    results: dict = {}
    if "openaq" in wanted:
        _run("openaq", lambda: openaq.download_hours(s, cities, start, end), results)
    if "firms" in wanted:
        _run("firms", lambda: firms.download(s, regions, start, end), results)
    if "weather" in wanted:
        _run("weather", lambda: openmeteo.download_weather(s, cities, start, end), results)
    if "cams" in wanted:
        _run("cams", lambda: openmeteo.download_cams(s, cities, start, end), results)
    return results


def cmd_update(s: cfg.Settings, args) -> dict:
    cities = s.pick(args.cities)
    today = date.today()
    regions = sorted({c.fire_region for c in cities})
    results: dict = {}
    _run("forecasts", lambda: openmeteo.snapshot(s, cities), results)   # first: these can't be fetched later
    _run("openaq", lambda: openaq.update_recent(s, cities), results)
    _run("firms", lambda: firms.download(s, regions, today - timedelta(days=12), today), results)
    # History: only the current month (and last month, until it settles), and only when the saved copy is stale.
    _run("cams", lambda: openmeteo.download_cams(s, cities, today - timedelta(days=7), today,
                                                 refresh_hours=openmeteo.REFRESH_HOURS["cams"]), results)
    _run("weather", lambda: openmeteo.download_weather(s, cities, today - timedelta(days=14), today,
                                                       refresh_hours=openmeteo.REFRESH_HOURS["weather"]), results)
    return results


def cmd_snapshot(s: cfg.Settings, args) -> dict:
    results: dict = {}
    _run("forecasts", lambda: openmeteo.snapshot(s, s.pick(args.cities)), results)
    return results


def cmd_status(s: cfg.Settings, args) -> dict:
    """Files, rows and newest timestamp per source; reads only metadata-light columns."""
    d = s.data_dir
    out = {}
    for name, pattern, col in (("openaq", "openaq/hours/**/*.parquet", "hour_utc"),
                               ("firms", "firms/**/*.parquet", "detected_utc"),
                               ("weather", "weather/history/**/*.parquet", "time_utc"),
                               ("cams", "cams/history/**/*.parquet", "time_utc"),
                               ("forecasts", "forecasts/**/*.parquet", "issued_utc")):
        files = sorted(d.glob(pattern))
        rows, newest = 0, None
        for p in files:
            df = pd.read_parquet(p, columns=[col])
            rows += len(df)
            if len(df):
                m = pd.to_datetime(df[col], utc=True).max()
                newest = m if newest is None or m > newest else newest
        out[name] = {"files": len(files), "rows": rows, "newest": str(newest) if newest is not None else None}
    sensors = d / "openaq" / "sensors.parquet"
    out["openaq_sensors"] = len(pd.read_parquet(sensors)) if sensors.exists() else 0
    for k, v in out.items():
        print(f"{k:16} {v}")
    return out


def _date(text: str) -> date:
    return date.fromisoformat(text)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m aqcast.sync", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("sensors", "backfill", "update", "snapshot", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("--cities", type=lambda t: [x.strip() for x in t.split(",") if x.strip()],
                        help="comma-separated city slugs (default: all in config/cities.yaml)")
        if name == "backfill":
            sp.add_argument("--source", choices=(*SOURCES, "all"), default="all")
            sp.add_argument("--start", type=_date, help=f"YYYY-MM-DD (default {cfg.HISTORY_START})")
            sp.add_argument("--end", type=_date, help="YYYY-MM-DD (default today)")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    args = parser().parse_args(argv)
    s = cfg.load()
    started = utc_now()
    results = {"sensors": cmd_sensors, "backfill": cmd_backfill, "update": cmd_update,
               "snapshot": cmd_snapshot, "status": cmd_status}[args.command](s, args)
    if args.command != "status":
        record = {"command": args.command, "started_utc": started.isoformat(), "finished_utc": utc_now().isoformat(),
                  "results": results}
        runs = Path(s.data_dir) / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{started:%Y%m%dT%H%M%S}_{args.command}.json").write_text(json.dumps(record, indent=2, default=str))
        failed = [k for k, v in results.items() if not v.get("ok") and not v.get("skipped")]
        if failed:
            log.error("Finished with failures: %s", ", ".join(failed))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
