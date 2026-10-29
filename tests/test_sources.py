from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from aqcast.sources import firms, openaq, openmeteo
from aqcast.settings import City
from aqcast.web import SourceError
from tests.fakes import FakeFetcher, FakeResponse

CITY = City("Testville", "testville", "XX", 28.6, 77.2, "south_asia")
NOW = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)


def iso(dt):
    return {"utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ")}


# ------------------------------------------------------------------------------------------------ OpenAQ
def location(lid, name, sensor_ids, monitor, first_days_ago, last_days_ago, lat=28.6, lon=77.2):
    return {"id": lid, "name": name, "isMonitor": monitor, "provider": {"name": "p"},
            "coordinates": {"latitude": lat, "longitude": lon},
            "datetimeFirst": iso(NOW - timedelta(days=first_days_ago)),
            "datetimeLast": iso(NOW - timedelta(days=last_days_ago)),
            "sensors": [{"id": i, "parameter": {"id": 2}} for i in sensor_ids] + [{"id": 9999, "parameter": {"id": 1}}]}


def test_a_station_keeps_all_its_records_and_sensors():
    # Like OpenAQ's Delhi data: one station listed as an old record and a current record with two sensor ids.
    locs = [location(235, "Anand Vihar, New Delhi", [12235610, 384], True, 3900, 0, 28.6469, 77.3152),
            location(5509, "Anand Vihar, Delhi", [14521], True, 3100, 1460, 28.6470, 77.3153),   # same spot, retired
            location(7, "Cheap sensor", [70], False, 4000, 0, 28.50, 77.10),
            location(8, "New monitor", [80], True, 200, 0, 28.70, 77.20),
            location(9, "Dead monitor", [90], True, 5000, 60, 28.55, 77.25)]
    f = FakeFetcher(lambda url, p: FakeResponse({"results": locs}))
    ranked = openaq.rank_sites(openaq.candidate_sensors(f, CITY), NOW, keep=2)
    assert set(ranked["sensor_id"]) == {12235610, 384, 14521, 80}       # whole Anand Vihar site, then New monitor
    assert ranked.loc[ranked["sensor_id"] == 14521, "site"].iloc[0] == "Anand Vihar, New Delhi"
    assert ranked["site_rank"].tolist() == [1, 1, 1, 2]
    assert 9999 not in set(ranked["sensor_id"])                          # PM10 ignored


def test_hours_filtered_to_window_and_paged():
    start, end = datetime(2026, 9, 1, tzinfo=timezone.utc), datetime(2026, 9, 2, tzinfo=timezone.utc)

    def answer(url, p):
        # Pretend the API ignores the date filter: page 1 is full (1000 rows), page 2 holds the rest.
        base = start - timedelta(hours=1000)
        hours = [base + timedelta(hours=i) for i in range(1030)]
        page = hours[(p["page"] - 1) * 1000: p["page"] * 1000]
        return FakeResponse({"results": [{"value": 10.0, "period": {"datetimeFrom": iso(h)},
                                          "coverage": {"percentComplete": 100}} for h in page]})

    f = FakeFetcher(answer)
    df = openaq.fetch_hours(f, 7, start, end)
    assert len(df) == 24 and df["hour_utc"].min() == pd.Timestamp(start)
    assert [c[1]["page"] for c in f.calls] == [1, 2]


# ------------------------------------------------------------------------------------------------- FIRMS
AVAIL = {"VIIRS_SNPP_SP": (date(2012, 1, 20), date(2026, 6, 25)),     # archive ends mid-window
         "VIIRS_SNPP_NRT": (date(2026, 6, 26), date(2026, 10, 28))}


def test_grid_is_fixed_whatever_the_start():
    assert firms.grid(date(2026, 6, 25), date(2026, 7, 10))[0] == firms.grid(date(2026, 6, 21), date(2026, 7, 10))[0]


def test_window_spanning_archive_end_splits_into_sp_and_nrt():
    w = [g for g in firms.grid(date(2026, 6, 1), date(2026, 7, 31)) if g <= date(2026, 6, 25) < g + timedelta(days=10)][0]
    parts = firms.plan_parts(w, "VIIRS_SNPP", AVAIL)
    assert [p[2] for p in parts] == ["VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"]
    assert sum(p[1] for p in parts) == 10
    assert parts[1][0] == date(2026, 6, 26)


def test_recent_window_is_cut_only_by_availability():
    w = firms.grid(date(2026, 10, 25), date(2026, 10, 28))[0]          # window still running
    parts = firms.plan_parts(w, "VIIRS_SNPP", AVAIL)
    assert parts == [(w, (date(2026, 10, 28) - w).days + 1, "VIIRS_SNPP_NRT")]


def test_firms_csv_parsed_and_error_text_rejected():
    csv = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,"
           "bright_ti5,frp,daynight\n30.1,75.2,330.1,0.4,0.4,2026-10-20,812,N,VIIRS,n,2.0NRT,290.1,5.3,D\n")
    df = firms.parse(csv)
    assert df.loc[0, "detected_utc"] == pd.Timestamp("2026-10-20 08:12", tz="UTC")
    assert df.loc[0, "frp"] == 5.3
    with pytest.raises(SourceError):
        firms.parse("Invalid MAP_KEY.")


# --------------------------------------------------------------------------------------------- Open-Meteo
def test_hourly_frame_and_errors():
    payload = {"hourly": {"time": ["2026-10-29T00:00", "2026-10-29T01:00"], "pm2_5": [41.2, None]}}
    df = openmeteo.hourly_frame(payload, ["pm2_5", "dust"])
    assert df["time_utc"].iloc[0] == pd.Timestamp("2026-10-29", tz="UTC")
    assert pd.isna(df["pm2_5"].iloc[1]) and df["dust"].isna().all()
    with pytest.raises(SourceError, match="Cannot"):
        openmeteo.hourly_frame({"error": True, "reason": "Cannot initialize"}, ["pm2_5"])


def test_open_meteo_request_weight():
    assert openmeteo.weight(4, 9) == 1          # a forecast: one call
    assert openmeteo.weight(31, 9) == 3         # a month of history: three calls
    assert openmeteo.weight(31, 12) == 6        # more than 10 variables doubles it


def test_snapshot_slot_is_six_hourly(monkeypatch):
    monkeypatch.setattr(openmeteo, "utc_now", lambda: datetime(2026, 10, 29, 17, 42, tzinfo=timezone.utc))
    assert openmeteo.issue_time() == datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
