"""Parquet files on disk: atomic writes, reads, and the date helpers used to split downloads into chunks."""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def write_frame(df: pd.DataFrame, path: Path) -> Path:
    """Write via a temporary file and rename, so a stopped container never leaves half a file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def fetched_after(path: Path, moment: datetime) -> bool:
    """True if the file exists and its rows were downloaded after `moment` (so the period was already over)."""
    if not path.exists():
        return False
    df = read_frame(path)
    if df.empty or "fetched_utc" not in df:
        return path.stat().st_mtime > moment.timestamp()
    return pd.to_datetime(df["fetched_utc"], utc=True).max() > pd.Timestamp(moment)


def month_starts(start: date, end: date) -> list[date]:
    out, d = [], date(start.year, start.month, 1)
    while d <= end:
        out.append(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


def month_end(d: date) -> date:
    nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return nxt - timedelta(days=1)


def chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    """Split [start, end] into consecutive ranges of at most `days` days."""
    out, d = [], start
    while d <= end:
        stop = min(d + timedelta(days=days - 1), end)
        out.append((d, stop))
        d = stop + timedelta(days=1)
    return out


def as_utc_midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
