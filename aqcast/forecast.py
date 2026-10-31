"""The live forecast: PM2.5 for every city 24, 48 and 72 hours ahead, from the latest data, with its drivers.

Run after `sync update` (every 6 hours in production):

    python -m aqcast.forecast

For each city it builds the same features the model was trained on, at the latest 6-hour issue time for which
the city has recent PM2.5:
  PM2.5     the last 10 days of sensor readings, through the same checks as the training data
  weather   what happened (reanalysis, about 5 days behind), then the first 6 hours of each saved forecast
            (close to what happened), then the latest forecast for the hours ahead
  CAMS      the CAMS history (refreshed every 6 hours by update), topped up with the latest saved forecast
  fires     the last 15 days of FIRMS detections, without the permanent hotspots found in training

Outputs:
  data/forecasts/model/issued=<YYYY-MM-DDTHH>/forecast.parquet   every forecast, kept for later scoring
  data/forecasts/model/latest.json                               what the API serves

Each forecast carries the model's value, the bias-corrected value (see model.py), an indicative US AQI
category, persistence and the CAMS forecast for comparison, the share of each feature group in it, and the
features that moved it most, as a percentage up or down.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from aqcast import clean, features, fires
from aqcast import settings as cfg
from aqcast.model import cams_baseline, cams_forecasts, design, group_of, model_dir
from aqcast.settings import City, Settings
from aqcast.storage import as_utc, month_starts, read_frame, utc_now, write_frame

log = logging.getLogger("aqcast")

PM_DAYS = 10
FIRE_DAYS = 15
MAX_STALENESS_HOURS = 24          # a city whose newest PM2.5 is older than this gets no forecast
TOP_FEATURES = 5
# US EPA PM2.5 categories (2024 breakpoints, ug/m3). They are defined for 24-hour means, so for a 6-hour
# forecast they are indicative only.
BANDS = [(9.0, "Good"), (35.4, "Moderate"), (55.4, "Unhealthy for sensitive groups"), (125.4, "Unhealthy"),
         (225.4, "Very unhealthy"), (float("inf"), "Hazardous")]


def band(pm25: float) -> str | None:
    if pm25 is None or not np.isfinite(pm25):
        return None
    for top, name in BANDS:
        if pm25 <= top:
            return name
    return None


# ------------------------------------------------------------------------------------------------- inputs
def recent_pm(settings: Settings, now: pd.Timestamp, days: int = PM_DAYS) -> pd.DataFrame:
    """Checked hourly city PM2.5 for the last `days` days: city, hour_utc, pm25."""
    d = settings.data_dir
    sensors = clean.usable_sensors(read_frame(d / "openaq" / "sensors.parquet"), clean.load_exclusions())
    used = sensors[sensors["excluded"] == ""]
    start = now - pd.Timedelta(days=days)
    months = [f"{m:%Y-%m}" for m in month_starts(start.date(), now.date())]
    parts = []
    for sid in used["sensor_id"]:
        for m in months:
            p = d / "openaq" / "hours" / f"sensor={sid}" / f"{m}.parquet"
            if p.exists():
                part = read_frame(p, columns=["sensor_id", "hour_utc", "pm25", "pct_complete"])
                if len(part):
                    part["pm25"] = pd.to_numeric(part["pm25"], errors="coerce").astype(float)
                    part["pct_complete"] = pd.to_numeric(part["pct_complete"], errors="coerce").astype(float)
                    parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["city", "hour_utc", "pm25"])
    readings = clean.flag_readings(clean.align_hours(pd.concat(parts, ignore_index=True)))
    series = clean.city_series(readings, used)
    return series[series["hour_utc"] >= start]


def _snapshots(settings: Settings, kind: str, city: City, since: pd.Timestamp) -> list[pd.DataFrame]:
    out = []
    for folder in sorted((settings.data_dir / "forecasts" / kind).glob("issued=*")):
        issued = pd.Timestamp(folder.name.split("=", 1)[1] + ":00", tz="UTC")
        p = folder / f"{city.slug}.parquet"
        if issued >= since and p.exists():
            df = read_frame(p)
            df["time_utc"] = as_utc(df["time_utc"])
            df["issued_utc"] = issued
            out.append(df)
    return out


def stitched(settings: Settings, kind: str, city: City, columns: list[str], now: pd.Timestamp) -> pd.DataFrame:
    """History, then the first 6 hours of each saved forecast, then the latest forecast; later sources win."""
    hist = features._history(settings, kind, city, columns).reset_index().assign(priority=0)
    snaps = [s for s in _snapshots(settings, kind, city, now - pd.Timedelta(days=PM_DAYS + 3))
             if s["issued_utc"].iloc[0] <= now]
    pieces = [hist]
    for s in snaps:
        lead = (s["time_utc"] - s["issued_utc"]) / pd.Timedelta(hours=1)
        pieces.append(s[lead < features.ISSUE_EVERY].assign(priority=1))
    if snaps:
        pieces.append(snaps[-1].assign(priority=2))
    df = pd.concat([p[["time_utc", "priority", *[c for c in columns if c in p]]] for p in pieces if len(p)],
                   ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC", name="time_utc"), dtype=float)
    df = df.sort_values(["time_utc", "priority"]).drop_duplicates("time_utc", keep="last")
    return df.drop(columns="priority").set_index("time_utc")


def recent_fires(settings: Settings, now: pd.Timestamp) -> pd.DataFrame:
    """Fire power around each city over the last FIRE_DAYS days, as in fires.py."""
    static_path = settings.data_dir / "clean" / "fire_static_cells.parquet"
    static = (np.sort(read_frame(static_path)["cell"].to_numpy(dtype=np.int64)) if static_path.exists()
              else np.array([], dtype=np.int64))
    first = (now - pd.Timedelta(days=FIRE_DAYS)).date()
    parts = []
    for region in sorted({c.fire_region for c in settings.cities}):
        cities = [c for c in settings.cities if c.fire_region == region]
        for p in fires.fire_files(settings, region):
            if date.fromisoformat(p.name[:10]) < first - timedelta(days=5):
                continue
            df = fires.keep_real_fires(fires.read_fires(p), static)
            for c in cities:
                r = fires.city_rings(df, c.slug, c.lat, c.lon)
                if len(r):
                    parts.append(r)
    cols = ["city", "period_utc", "ring_km", "sector", "frp", "n"]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True).groupby(cols[:4], as_index=False)[["frp", "n"]].sum()


# ------------------------------------------------------------------------------------------------- model
def load_model(directory: Path):
    import lightgbm
    card = json.loads((directory / "model_card.json").read_text(encoding="utf-8"))
    return lightgbm.Booster(model_file=str(directory / "pm25_lgbm.txt")), card


def explain(contrib: np.ndarray, feature_names: list[str]) -> tuple[dict, list[dict]]:
    """Group shares (% of |SHAP|) and the strongest features as % change of the forecast."""
    c = pd.Series(contrib[:-1], index=feature_names)
    groups = c.abs().groupby(lambda f: group_of(f)).sum()
    shares = (groups / groups.sum() * 100).round(1) if groups.sum() > 0 else groups
    top = c.reindex(c.abs().sort_values(ascending=False).index[:TOP_FEATURES])
    return ({k: float(v) for k, v in shares.sort_values(ascending=False).items()},
            [{"feature": f, "group": group_of(f), "effect_pct": round(float(np.expm1(v) * 100), 1)}
             for f, v in top.items()])


def latest_rows(settings: Settings, now: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """One row per city and horizon at each city's latest issue time, plus what each city's PM2.5 looks like."""
    pm_all = recent_pm(settings, now)
    fire_tab = recent_fires(settings, now)
    rows, observed, skipped = [], {}, []
    for city in settings.cities:
        pm = pm_all[pm_all["city"] == city.slug].set_index("hour_utc")["pm25"].sort_index()
        if pm.empty or (now - pm.index.max()) > pd.Timedelta(hours=MAX_STALENESS_HOURS):
            skipped.append(city.slug)
            continue
        observed[city.slug] = {"last_hour_utc": pm.index.max().isoformat(),
                               "pm25_last_3h": round(float(pm.tail(3).mean()), 1),
                               "pm25_last_24h": round(float(pm[pm.index > pm.index.max() - pd.Timedelta(hours=24)].mean()), 1)}
        wx = stitched(settings, "weather", city, features.WX, now)
        cams = stitched(settings, "cams", city, list(features.CAMS), now)
        t = features.city_table(city, pm, wx, cams, fire_tab[fire_tab["city"] == city.slug],
                                labeled_only=False, until=now + pd.Timedelta(hours=max(features.HORIZONS) + 1))
        if t.empty:
            continue
        rows.append(t[t["issue_utc"] == t["issue_utc"].max()])
    if skipped:
        log.warning("No PM2.5 in the last %d hours, so no forecast for: %s (run: sync update)",
                    MAX_STALENESS_HOURS, ", ".join(skipped))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), observed


def run(settings: Settings, now: pd.Timestamp | None = None, booster=None, card=None) -> dict:
    now = pd.Timestamp(now) if now is not None else pd.Timestamp(utc_now())
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    if booster is None:
        booster, card = load_model(model_dir(settings))
    feats, cities = card["features"], card["cities"]
    rows, observed = latest_rows(settings, now)
    if rows.empty:
        raise RuntimeError("No city has recent PM2.5. Run:  sync update")
    rows = rows[rows["city"].isin(cities)].reset_index(drop=True)
    X = design(rows, feats, cities)
    rows["pm25"] = np.expm1(booster.predict(X)).clip(min=0)
    rows["pm25_adjusted"] = rows["pm25"] * rows["city"].map(card.get("bias_factors_by_city", {})).fillna(1.0)
    rows["persistence"] = rows["pm_24h"].fillna(rows["pm_3h"])
    rows["cams"] = cams_baseline(rows, cams_forecasts(settings))
    contrib = np.asarray(booster.predict(X, pred_contrib=True))

    by_slug = {c.slug: c for c in settings.cities}
    out_rows, payload = [], []
    for slug, g in rows.groupby("city", sort=False):
        c = by_slug[slug]
        items = []
        for i, r in g.sort_values("horizon_h").iterrows():
            shares, top = explain(contrib[i], feats)
            item = {"horizon_h": int(r["horizon_h"]), "target_utc": r["target_utc"].isoformat(),
                    "pm25": round(float(r["pm25"]), 1), "pm25_adjusted": round(float(r["pm25_adjusted"]), 1),
                    "category": band(float(r["pm25_adjusted"])),
                    "persistence": None if pd.isna(r["persistence"]) else round(float(r["persistence"]), 1),
                    "cams": None if pd.isna(r["cams"]) else round(float(r["cams"]), 1),
                    "drivers_pct": shares, "top_features": top}
            items.append(item)
            out_rows.append({"city": slug, "issue_utc": r["issue_utc"], **{k: v for k, v in item.items()
                             if k not in ("drivers_pct", "top_features")},
                             "drivers_pct": json.dumps(shares), "top_features": json.dumps(top)})
        payload.append({"city": slug, "name": c.name, "country": c.country, "lat": c.lat, "lon": c.lon,
                        "known_for": c.driver, "issue_utc": g["issue_utc"].iloc[0].isoformat(),
                        "observed": observed.get(slug), "forecasts": items})

    issued = rows["issue_utc"].max()
    base = settings.data_dir / "forecasts" / "model"
    write_frame(pd.DataFrame(out_rows), base / f"issued={issued:%Y-%m-%dT%H}" / "forecast.parquet")
    doc = {"made_utc": now.isoformat(), "issue_utc": issued.isoformat(),
           "model": {"trained_from": card.get("data_from"), "trained_to": card.get("data_to"),
                     "scores": card.get("scores")},
           "cities": payload}
    tmp = base / "latest.json.tmp"
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    tmp.replace(base / "latest.json")
    for p in payload:
        f = {x["horizon_h"]: x["pm25_adjusted"] for x in p["forecasts"]}
        log.info("%-14s now %6s  24h %6s  48h %6s  72h %6s", p["name"], (p["observed"] or {}).get("pm25_last_3h"),
                 f.get(24), f.get(48), f.get(72))
    return {"cities": len(payload), "issue_utc": issued.isoformat(), "file": str(base / "latest.json")}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        result = run(cfg.load())
    except (RuntimeError, FileNotFoundError) as e:
        log.error("forecast: %s", e)
        return 1
    log.info("forecast: %s", json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
