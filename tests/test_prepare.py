"""Data checks, fire processing and the training table, on small made-up data."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from aqcast import clean, features, fires
from aqcast.settings import City

T0 = pd.Timestamp("2026-01-01", tz="UTC")
CITY = City("Testville", "testville", "XX", 28.6, 77.2, "south_asia")


def hours(n, start=T0):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def readings(values, sensor=1, coverage=100.0):
    return pd.DataFrame({"sensor_id": sensor, "hour_utc": hours(len(values)), "pm25": values,
                         "pct_complete": coverage})


# ------------------------------------------------------------------------------------------------ checks
def test_each_rule_flags_its_case():
    rng = np.random.default_rng(1)
    vals = list(40 + rng.normal(0, 3, 60))
    vals[5] = None            # missing
    vals[10] = -20            # negative
    vals[11] = -2             # small negative: set to 0
    vals[20] = 2000           # too high
    vals[30] = 400            # spike
    df = readings(vals)
    df.loc[40, "pct_complete"] = 50
    out = clean.flag_readings(df).set_index(df["hour_utc"])
    f = out["flag"].tolist()
    assert (f[5], f[10], f[20], f[30], f[40]) == ("missing", "negative", "too_high", "spike", "low_coverage")
    assert f[11] == "ok" and out["pm25"].iloc[11] == 0
    assert f.count("ok") == 60 - 5


def test_stuck_sensor_is_flagged_but_a_short_repeat_is_not():
    vals = [30.0] * 30 + [31.0, 32.0] + [33.0] * 5 + [34.0]
    out = clean.flag_readings(readings(vals))
    assert (out["flag"].iloc[:30] == "flatline").all()
    assert (out["flag"].iloc[30:] == "ok").all()


def test_a_gap_breaks_a_flatline():
    df = readings([30.0] * 30)
    df = df.drop(index=15)                      # one hour missing: two runs of 15 and 14
    assert (clean.flag_readings(df)["flag"] == "ok").all()


def test_indoor_test_and_listed_sensors_are_excluded():
    s = pd.DataFrame({"sensor_id": [1, 2, 3, 4], "city": "x",
                      "location": ["Robinson's Tower (Indoor)", "13V_Test", "Civic", "2-1"]})
    out = clean.usable_sensors(s, {4: "unnamed one-month unit"})
    assert out["excluded"].tolist()[2] == ""
    assert all(out["excluded"].tolist()[i] for i in (0, 1, 3))


def test_half_hour_stamps_move_to_the_next_whole_hour():
    df = pd.DataFrame({"sensor_id": 1, "hour_utc": pd.to_datetime(["2026-01-01 05:30", "2026-01-01 06:15",
                                                                    "2026-01-01 08:00"], utc=True),
                       "pm25": [1.0, 2.0, 3.0], "pct_complete": 100.0})
    out = clean.align_hours(df)
    assert [t.hour for t in out["hour_utc"]] == [6, 7, 8]
    assert all(t.minute == 0 for t in out["hour_utc"])


def test_coverage_figure_ignored_for_a_sensor_that_always_reports_it_low():
    always_low = readings([40.0 + i % 5 for i in range(48)], sensor=1, coverage=50.0)
    normal = readings([40.0 + i % 5 for i in range(48)], sensor=2)
    normal.loc[10, "pct_complete"] = 50.0
    out = clean.flag_readings(pd.concat([always_low, normal], ignore_index=True))
    assert (out.loc[out["sensor_id"] == 1, "flag"] == "ok").all()
    assert (out.loc[out["sensor_id"] == 2, "flag"] == "low_coverage").sum() == 1


def test_station_then_city_medians():
    r = pd.concat([readings([10.0] * 3, sensor=1), readings([30.0] * 3, sensor=2),   # one station, 2 records
                   readings([100.0] * 3, sensor=3), readings([50.0] * 3, sensor=4)])
    r["flag"] = "ok"
    sensors = pd.DataFrame({"sensor_id": [1, 2, 3, 4], "city": "x", "site_id": [7, 7, 8, 9]})
    out = clean.city_series(r, sensors)
    assert out["pm25"].tolist() == [50.0] * 3      # median of stations 20, 100, 50
    assert out["n_sites"].tolist() == [3] * 3


# ------------------------------------------------------------------------------------------------- fires
def test_bearing_and_sector():
    dist, bearing = fires.distance_bearing(0.0, 0.0, [1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0])
    assert np.allclose(dist, 111.2, atol=0.5)
    assert np.allclose(bearing, [0, 90, 180, 270], atol=0.01)
    assert fires.sector_of(bearing).tolist() == [0, 2, 4, 6]
    assert fires.sector_of([350.0, 22.0, 23.0]).tolist() == [0, 0, 1]


def test_permanent_hotspots_found():
    months = pd.date_range("2023-01-15", periods=24, freq="MS", tz="UTC") + pd.Timedelta(days=14)
    flare = pd.DataFrame({"latitude": 30.001, "longitude": 70.001, "frp": 5.0, "detected_utc": months})
    crop = pd.DataFrame({"latitude": 31.0, "longitude": 75.0, "frp": 50.0, "detected_utc": months[[9, 10, 21]]})
    # Savanna that burns every dry season (Nov-Mar) for 4 years: many months, but never all year.
    dry = pd.DatetimeIndex([pd.Timestamp(f"{y}-{m:02d}-10", tz="UTC") for y in range(2022, 2027)
                            for m in (1, 2, 3, 11, 12)])
    savanna = pd.DataFrame({"latitude": 5.0, "longitude": 20.0, "frp": 30.0, "detected_utc": dry})
    out = fires.static_cells([flare, crop, savanna])
    assert len(out) == 1
    assert abs(out["lat"].iloc[0] - 30.01) < 0.02 and abs(out["lon"].iloc[0] - 70.01) < 0.02


def test_rings_sectors_and_periods():
    df = pd.DataFrame({
        "latitude": [28.6, 28.6, 30.0, 40.0],
        "longitude": [77.7, 77.7, 77.2, 77.2],        # ~49 km east (twice), ~156 km north, too far
        "frp": [10.0, 5.0, 20.0, 99.0],
        "detected_utc": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 05:59", "2026-01-01 07:00",
                                        "2026-01-01 01:00"], utc=True)})
    out = fires.city_rings(df, "testville", 28.6, 77.2)
    east = out[(out["sector"] == 2) & (out["ring_km"] == 0)]
    assert east["frp"].tolist() == [15.0] and east["n"].tolist() == [2]
    assert east["period_utc"].iloc[0] == pd.Timestamp("2026-01-01 00:00", tz="UTC")
    north = out[out["sector"] == 0]
    assert north["ring_km"].tolist() == [100] and north["period_utc"].iloc[0].hour == 6
    assert out["frp"].sum() == 35.0


def test_low_confidence_and_static_fires_dropped():
    df = pd.DataFrame({"latitude": [30.001, 31.0, 32.0], "longitude": [70.001, 75.0, 76.0],
                       "frp": 1.0, "confidence": ["n", "l", "h"],
                       "detected_utc": pd.Timestamp("2026-01-01", tz="UTC")})
    static = fires.cell_key([30.001], [70.001])
    assert fires.keep_real_fires(df, static)["latitude"].tolist() == [32.0]


# ----------------------------------------------------------------------------------------------- features
def test_upwind_weights():
    w = features.upwind_weights(np.array([90.0]))          # wind from the east
    assert w[0, 2] == 1.0                                   # fires to the east are upwind
    assert w[0, 6] == 0.0                                   # fires to the west are downwind
    assert np.isclose(w[0, 1], np.cos(np.radians(45)))


def test_training_rows_use_only_the_past_and_target_the_future():
    n = 24 * 6
    pm = pd.Series(np.arange(n, dtype=float), index=hours(n))   # value = hours since start
    wx = pd.DataFrame({"temperature_2m": 20.0, "wind_speed_10m": 3.0, "wind_direction_10m": 90.0,
                       "precipitation": 0.0}, index=hours(n))
    cams = pd.DataFrame({"pm2_5": 1.0}, index=hours(n))
    fires_tab = pd.DataFrame({"city": "testville", "period_utc": [T0 + pd.Timedelta(hours=18)],
                              "ring_km": [0], "sector": [2], "frp": [100.0], "n": [3]})
    out = features.city_table(CITY, pm, wx, cams, fires_tab)
    row = out[(out["issue_utc"] == T0 + pd.Timedelta(hours=48)) & (out["horizon_h"] == 24)].iloc[0]
    # Known at 48:00 are hours starting up to 47:00, so the last 3 hours are 45, 46, 47.
    assert row["pm_3h"] == 46.0
    # Target: hours starting 66:00-71:00 (the six hours ending at 72:00).
    assert row["y"] == np.mean([66, 67, 68, 69, 70, 71])
    assert row["target_utc"] == T0 + pd.Timedelta(hours=72)
    # The fire period 18:00-24:00 ended before 48:00 but more than 24 h earlier: only in the 72-hour sums.
    assert row["fire_frp_24h_r0"] == 0.0 and row["fire_frp_72h_r0"] == 100.0
    early = out[(out["issue_utc"] == T0 + pd.Timedelta(hours=36)) & (out["horizon_h"] == 24)].iloc[0]
    assert early["fire_frp_24h_r0"] == 100.0 and early["fire_upwind_24h_r0"] == 100.0   # east wind, east fire
    at_fire = out[(out["issue_utc"] == T0 + pd.Timedelta(hours=18)) & (out["horizon_h"] == 24)].iloc[0]
    assert at_fire["fire_frp_24h_r0"] == 0.0                # that period hasn't ended yet at 18:00
    assert set(out["horizon_h"]) == {24, 48, 72} and out["y"].notna().all()


def test_no_fires_means_zero_fire_features():
    n = 24 * 4
    pm = pd.Series(30.0, index=hours(n))
    empty = pd.DataFrame(columns=["city", "period_utc", "ring_km", "sector", "frp", "n"])
    out = features.city_table(CITY, pm, pd.DataFrame(index=hours(n)), pd.DataFrame(index=hours(n)), empty)
    assert len(out) and (out.filter(like="fire_") == 0).all().all()
