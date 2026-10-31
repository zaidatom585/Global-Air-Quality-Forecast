"""HTTP API for the forecasts: a small FastAPI app that serves what `aqcast.forecast` saved.

    uvicorn aqcast.api:app --host 0.0.0.0 --port 8000

Endpoints (interactive docs at /docs):
  GET /health              is a forecast available, and how old is it
  GET /cities              the cities, with their latest observed PM2.5
  GET /forecast            the latest forecast for every city
  GET /forecast/{city}     one city: 24, 48 and 72 hours ahead, with drivers
  GET /model               how the model was trained and how it scored

The app only reads files (data/forecasts/model/latest.json and models/), so it starts in a second, needs no
model library, and can run as several copies behind a load balancer.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

from aqcast import settings as cfg

DATA_DIR = Path(os.environ.get("AQ_DATA_DIR", cfg.DATA_DIR))
MODEL_DIR = Path(os.environ.get("AQ_MODEL_DIR", DATA_DIR.parent / "models"))
STALE_HOURS = 12

app = FastAPI(
    title="Global Air Quality Forecast",
    description="PM2.5 forecasts 24-72 hours ahead for 36 cities, from ground sensors, satellite fire "
                "detections, weather and the CAMS forecast. Indicative only; not an official air-quality service.",
    version="0.4.0",
)

_cache: dict[Path, tuple[float, object]] = {}


def _read_json(path: Path):
    """Read a JSON file, re-reading only when it changes on disk."""
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    doc = json.loads(path.read_text(encoding="utf-8"))
    _cache[path] = (mtime, doc)
    return doc


def latest() -> dict:
    doc = _read_json(DATA_DIR / "forecasts" / "model" / "latest.json")
    if doc is None:
        raise HTTPException(503, "No forecast yet. Run the forecast job: python -m aqcast.forecast")
    return doc


def _age_hours(iso: str) -> float:
    t = datetime.fromisoformat(iso)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - t).total_seconds() / 3600, 1)


@app.get("/")
def root() -> dict:
    return {"service": app.title, "docs": "/docs",
            "endpoints": ["/health", "/cities", "/forecast", "/forecast/{city}", "/model"]}


@app.get("/health")
def health() -> dict:
    doc = _read_json(DATA_DIR / "forecasts" / "model" / "latest.json")
    if doc is None:
        return {"status": "no forecast yet"}
    age = _age_hours(doc["made_utc"])
    return {"status": "ok" if age <= STALE_HOURS else "stale", "forecast_made_utc": doc["made_utc"],
            "age_hours": age, "cities": len(doc["cities"])}


@app.get("/cities")
def cities() -> list[dict]:
    return [{k: c.get(k) for k in ("city", "name", "country", "lat", "lon", "known_for", "observed")}
            for c in latest()["cities"]]


@app.get("/forecast")
def forecast_all() -> dict:
    doc = latest()
    return {"made_utc": doc["made_utc"], "issue_utc": doc["issue_utc"],
            "cities": [{"city": c["city"], "name": c["name"], "issue_utc": c["issue_utc"],
                        "forecasts": [{k: f[k] for k in ("horizon_h", "target_utc", "pm25_adjusted", "category")}
                                      for f in c["forecasts"]]}
                       for c in doc["cities"]]}


@app.get("/forecast/{city}")
def forecast_city(city: str) -> dict:
    doc = latest()
    for c in doc["cities"]:
        if c["city"] == city.lower():
            return {"made_utc": doc["made_utc"], **c}
    known = ", ".join(c["city"] for c in doc["cities"])
    raise HTTPException(404, f"No forecast for '{city}'. Cities with a forecast: {known}")


@app.get("/model")
def model_info() -> dict:
    card = _read_json(MODEL_DIR / "model_card.json")
    if card is None:
        raise HTTPException(503, "No trained model found. Run: python -m aqcast.train")
    by_city = []
    metrics = MODEL_DIR / "report" / "metrics.csv"
    if metrics.exists():
        with metrics.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("level") == "city":
                    by_city.append({"city": r["city"], "mean_pm25": float(r["mean_pm25"]),
                                    "mae_model": float(r["mae_model"]),
                                    "mae_persistence": float(r["mae_persistence"]),
                                    "skill_vs_persistence": float(r["skill_vs_persistence"])})
    return {"model": card.get("model"), "trained_from": card.get("data_from"), "trained_to": card.get("data_to"),
            "rows": card.get("rows"), "features": len(card.get("features", [])), "scores": card.get("scores"),
            "scores_by_city": by_city}
