"""Live forecast helpers, and the API on a saved forecast file."""
import json

import numpy as np
import pandas as pd
import pytest

from aqcast import forecast


def test_us_aqi_categories():
    assert forecast.band(5) == "Good" and forecast.band(9.0) == "Good"
    assert forecast.band(20) == "Moderate" and forecast.band(300) == "Hazardous"
    assert forecast.band(float("nan")) is None


def test_explain_groups_and_top_features():
    contrib = np.array([0.2, -0.1, 0.05, 0.0, 3.0])          # last value is the base, not a feature
    shares, top = forecast.explain(contrib, ["fire_frp_24h_r100", "wx_wind", "pm_3h", "city"])
    assert shares["fires"] == pytest.approx(57.1, abs=0.1)
    assert top[0]["feature"] == "fire_frp_24h_r100" and top[0]["effect_pct"] == pytest.approx(22.1, abs=0.1)
    assert top[1]["effect_pct"] < 0


def test_later_weather_sources_win(tmp_path, monkeypatch):
    from aqcast.settings import City, Settings
    city = City("T", "t", "XX", 0.0, 0.0, "south_asia")
    s = Settings(cities=(city,), fire_regions={}, data_dir=tmp_path)
    now = pd.Timestamp("2026-10-31 13:00", tz="UTC")
    hours = pd.date_range("2026-10-30 00:00", periods=60, freq="h", tz="UTC")
    hist = pd.DataFrame({"temperature_2m": 1.0}, index=pd.DatetimeIndex(hours[:20], name="time_utc"))
    snap_old = pd.DataFrame({"time_utc": hours[24:60], "temperature_2m": 2.0,
                             "issued_utc": pd.Timestamp("2026-10-31 00:00", tz="UTC")})
    snap_new = pd.DataFrame({"time_utc": hours[36:60], "temperature_2m": 3.0,
                             "issued_utc": pd.Timestamp("2026-10-31 12:00", tz="UTC")})
    monkeypatch.setattr(forecast.features, "_history", lambda *a: hist)
    monkeypatch.setattr(forecast, "_snapshots", lambda *a: [snap_old, snap_new])
    out = forecast.stitched(s, "weather", city, ["temperature_2m"], now)["temperature_2m"]
    assert out.loc[hours[5]] == 1.0                      # history
    assert out.loc[hours[26]] == 2.0                     # first hours of the older forecast
    assert pd.Timestamp(hours[33]) not in out.index      # later hours of the older forecast are not used
    assert out.loc[hours[50]] == 3.0                     # the latest forecast for the hours ahead


# ---------------------------------------------------------------------------------------------------- API
def _api(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from aqcast import api
    monkeypatch.setattr(api, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path / "models")
    api._cache.clear()
    return TestClient(api.app)


def test_api_without_a_forecast_says_so(tmp_path, monkeypatch):
    client = _api(tmp_path, monkeypatch)
    assert client.get("/health").json()["status"] == "no forecast yet"
    assert client.get("/forecast").status_code == 503


def test_api_serves_the_latest_forecast(tmp_path, monkeypatch):
    client = _api(tmp_path, monkeypatch)
    doc = {"made_utc": pd.Timestamp.now(tz="UTC").isoformat(), "issue_utc": "2026-10-31T12:00:00+00:00",
           "model": {}, "cities": [{"city": "delhi", "name": "Delhi", "country": "IN", "lat": 28.6, "lon": 77.2,
                                    "known_for": "crop burning", "issue_utc": "2026-10-31T12:00:00+00:00",
                                    "observed": {"pm25_last_3h": 140.0},
                                    "forecasts": [{"horizon_h": 24, "target_utc": "2026-11-01T12:00:00+00:00",
                                                   "pm25": 130.0, "pm25_adjusted": 140.0, "category": "Very unhealthy",
                                                   "persistence": 145.0, "cams": 60.0, "drivers_pct": {"fires": 20.0},
                                                   "top_features": []}]}]}
    path = tmp_path / "data" / "forecasts" / "model" / "latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(doc))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/cities").json()[0]["city"] == "delhi"
    assert client.get("/forecast").json()["cities"][0]["forecasts"][0]["pm25_adjusted"] == 140.0
    one = client.get("/forecast/Delhi").json()
    assert one["forecasts"][0]["drivers_pct"]["fires"] == 20.0
    missing = client.get("/forecast/atlantis")
    assert missing.status_code == 404 and "delhi" in missing.json()["detail"]
    assert client.get("/model").status_code == 503
