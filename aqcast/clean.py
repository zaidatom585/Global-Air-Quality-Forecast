"""PM2.5 data checks, and the hourly city series the model learns from.

Steps:
1. Sensors: drop indoor, test, calibration and demo units (by name), and any listed in
   config/sensor_exclusions.yaml with the reason.
2. Readings: hours are aligned to whole UTC hours first. Stations in India, Nepal and a few other places
   report hours that start at :30 or :15 UTC (their local clock hours); each is moved to the next whole
   hour, so that it is never treated as known before it has ended. Then each hourly value gets a flag; only
   "ok" values are used.
     missing       no value
     low_coverage  the sensor reported for less than 75% of the hour (ignored for a sensor whose coverage
                   figure is below 75% most of the time: there the figure itself is wrong, not the data)
     negative      below -5 (values between -5 and 0 are noise around zero and are set to 0)
     too_high      above 1,500 ug/m3, beyond anything measured even in the worst smog
     flatline      the same value for 24 hours or more in a row: a stuck sensor
     spike         more than 3 times the sensor's 25-hour running median and over 100 above it: a glitch
3. Stations and cities: a station's value for an hour is the median of its sensors (old and new records of
   one station), and the city's value is the median of its stations, with the number of stations behind it.

Outputs:
  data/clean/pm25_hourly.parquet      city, hour_utc, pm25, n_sites, n_sensors
  data/clean/sensor_quality.parquet   per sensor: readings, share flagged by each rule, used or excluded
  data/clean/quality_report.json      per city: hours covered and share of readings dropped by each rule
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from aqcast.settings import ROOT, Settings
from aqcast.storage import as_utc, read_frame, write_frame

log = logging.getLogger(__name__)

EXCLUSIONS_FILE = ROOT / "config" / "sensor_exclusions.yaml"
NOT_OUTDOOR = re.compile(r"indoor|test|calib|demo", re.I)   # in a sensor's location name

MIN_COVERAGE = 75.0      # % of the hour the sensor reported
NEG_FLOOR = -5.0         # small negatives are instrument noise around zero
MAX_PM25 = 1500.0
FLAT_HOURS = 24
SPIKE_WINDOW = 25
SPIKE_RATIO = 3.0
SPIKE_ABOVE = 100.0
FLAGS = ["missing", "low_coverage", "negative", "too_high", "flatline", "spike"]


def load_exclusions(path: Path = EXCLUSIONS_FILE) -> dict[int, str]:
    """{sensor_id: reason} from config/sensor_exclusions.yaml."""
    if not Path(path).exists():
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {int(x["sensor_id"]): str(x.get("reason", "")) for x in raw.get("exclude", [])}


def usable_sensors(sensors: pd.DataFrame, exclusions: dict[int, str]) -> pd.DataFrame:
    """The sensor list with an `excluded` column holding the reason ('' when the sensor is used)."""
    df = sensors.copy()
    by_name = df["location"].fillna("").astype(str).str.contains(NOT_OUTDOOR)
    listed = df["sensor_id"].astype(int).map(exclusions)
    df["excluded"] = np.where(listed.notna(), listed.fillna(""),
                              np.where(by_name, "name says indoor, test, calibration or demo", ""))
    return df


def flag_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `flag` column ("ok" or the first rule a reading fails) to hourly readings of one or more sensors.
    Expects sensor_id, hour_utc, pm25 and pct_complete; small negatives are set to 0."""
    df = df.sort_values(["sensor_id", "hour_utc"]).reset_index(drop=True)
    pm = pd.to_numeric(df["pm25"], errors="coerce")
    cov = pd.to_numeric(df.get("pct_complete"), errors="coerce") if "pct_complete" in df else pd.Series(np.nan, index=df.index)
    flag = pd.Series("ok", index=df.index, dtype=object)
    flag[pm.isna()] = "missing"
    # Some sensors always report a low coverage figure (OpenAQ assumes a different reporting interval for
    # them). For those the figure says nothing about a single hour, so it is ignored.
    typical = cov.groupby(df["sensor_id"]).transform("median")
    trusted = typical >= MIN_COVERAGE
    flag[(flag == "ok") & trusted & cov.notna() & (cov < MIN_COVERAGE)] = "low_coverage"
    flag[(flag == "ok") & (pm < NEG_FLOOR)] = "negative"
    flag[(flag == "ok") & (pm > MAX_PM25)] = "too_high"
    pm = pm.where(~((flag == "ok") & (pm < 0)), 0.0)

    ok = flag == "ok"
    # Flatline: consecutive hours (no gap) with exactly the same value.
    sub = pd.DataFrame({"sensor_id": df["sensor_id"], "hour": df["hour_utc"], "pm": pm})[ok]
    new_run = (sub["sensor_id"].ne(sub["sensor_id"].shift())
               | sub["pm"].ne(sub["pm"].shift())
               | (sub["hour"] - sub["hour"].shift()).ne(pd.Timedelta(hours=1)))
    run_id = new_run.cumsum()
    run_len = run_id.map(run_id.value_counts())
    flag[sub.index[run_len.to_numpy() >= FLAT_HOURS]] = "flatline"

    # Spike: far above the sensor's own running median.
    ok = flag == "ok"
    sub = pd.DataFrame({"sensor_id": df["sensor_id"], "pm": pm})[ok]
    med = sub.groupby("sensor_id")["pm"].transform(
        lambda s: s.rolling(SPIKE_WINDOW, center=True, min_periods=5).median())
    spike = (sub["pm"] > SPIKE_RATIO * med) & (sub["pm"] - med > SPIKE_ABOVE)
    flag[sub.index[spike.to_numpy()]] = "spike"

    df["pm25"] = pm
    df["flag"] = flag
    return df


def align_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Move hour stamps that aren't on a whole UTC hour to the next whole hour; keep one value per sensor-hour."""
    df = df.copy()
    df["hour_utc"] = as_utc(df["hour_utc"]).dt.ceil("h")
    return df.drop_duplicates(["sensor_id", "hour_utc"], keep="last")


def city_series(readings: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    """Median of sensors per station and hour, then median of stations per city and hour."""
    ok = readings[readings["flag"] == "ok"].merge(sensors[["sensor_id", "city", "site_id"]], on="sensor_id")
    site = (ok.groupby(["city", "site_id", "hour_utc"])
              .agg(pm25=("pm25", "median"), n_sensors=("sensor_id", "nunique")).reset_index())
    city = (site.groupby(["city", "hour_utc"])
                .agg(pm25=("pm25", "median"), n_sites=("site_id", "nunique"), n_sensors=("n_sensors", "sum"))
                .reset_index())
    return city.sort_values(["city", "hour_utc"]).reset_index(drop=True)


def report(readings: pd.DataFrame, sensors: pd.DataFrame, series: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Per city summary and per sensor table."""
    r = readings.merge(sensors[["sensor_id", "city"]], on="sensor_id")
    per_sensor = (r.groupby("sensor_id")["flag"].value_counts(normalize=True).unstack(fill_value=0.0)
                    .reindex(columns=["ok", *FLAGS], fill_value=0.0).mul(100).round(1))
    per_sensor.insert(0, "readings", r.groupby("sensor_id").size())
    per_sensor = sensors[["city", "sensor_id", "site", "location", "excluded"]].merge(
        per_sensor.reset_index(), on="sensor_id", how="left")
    out = {}
    for city, g in r.groupby("city"):
        s = series[series["city"] == city]
        shares = g["flag"].value_counts(normalize=True).mul(100).round(2)
        if s.empty:
            out[city] = {"hours": 0, "readings": int(len(g)),
                         "dropped_pct": {k: float(shares.get(k, 0.0)) for k in FLAGS}}
            continue
        span = (s["hour_utc"].max() - s["hour_utc"].min()) / pd.Timedelta(hours=1) + 1
        out[city] = {
            "first": str(s["hour_utc"].min()), "last": str(s["hour_utc"].max()),
            "hours": int(len(s)), "coverage_pct": round(100 * len(s) / span, 1),
            "mean_pm25": round(float(s["pm25"].mean()), 1), "median_sites": float(s["n_sites"].median()),
            "dropped_pct": {k: float(shares.get(k, 0.0)) for k in FLAGS},
        }
    return out, per_sensor


def run(settings: Settings) -> dict:
    d = settings.data_dir
    sensors = usable_sensors(read_frame(d / "openaq" / "sensors.parquet"), load_exclusions())
    for row in sensors[sensors["excluded"] != ""].itertuples():
        log.info("excluded %-12s sensor %-9s %-36.36s %s", row.city, row.sensor_id, row.location, row.excluded)
    used = sensors[sensors["excluded"] == ""]
    parts = []
    for sid in used["sensor_id"]:
        for p in sorted((d / "openaq" / "hours" / f"sensor={sid}").glob("*.parquet")):
            part = read_frame(p, columns=["sensor_id", "hour_utc", "pm25", "pct_complete"])
            if len(part):
                # Fixed types, so an all-empty column in one month doesn't change the combined table's types.
                part["pm25"] = pd.to_numeric(part["pm25"], errors="coerce").astype(float)
                part["pct_complete"] = pd.to_numeric(part["pct_complete"], errors="coerce").astype(float)
                parts.append(part)
    if not parts:
        raise RuntimeError("No OpenAQ readings found. Run:  backfill --source openaq")
    readings = pd.concat(parts, ignore_index=True)
    readings = align_hours(readings)
    readings = flag_readings(readings)
    series = city_series(readings, used)
    summary, per_sensor = report(readings, sensors, series)

    out = d / "clean"
    write_frame(series, out / "pm25_hourly.parquet")
    write_frame(per_sensor, out / "sensor_quality.parquet")
    (out / "quality_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    header = f"{'city':14} {'hours':>7} {'coverage':>9} {'mean':>6} {'sites':>6}  dropped %: " + " ".join(FLAGS)
    log.info("%s", header)
    for city, s in summary.items():
        if not s.get("hours"):
            log.info("%-14s no usable readings out of %d (dropped %%: %s)", city, s.get("readings", 0),
                     " ".join(f"{s['dropped_pct'][k]:.1f}" for k in FLAGS))
            continue
        log.info("%-14s %7d %8.1f%% %6.1f %6.1f  %s", city, s["hours"], s["coverage_pct"], s["mean_pm25"],
                 s["median_sites"], " ".join(f"{s['dropped_pct'][k]:.1f}" for k in FLAGS))
    return {"readings": int(len(readings)), "ok_readings": int((readings["flag"] == "ok").sum()),
            "excluded_sensors": int((sensors["excluded"] != "").sum()), "city_hours": int(len(series)),
            "cities": int(series["city"].nunique())}
