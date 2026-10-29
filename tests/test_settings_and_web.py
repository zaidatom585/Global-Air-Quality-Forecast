import pytest

from aqcast import settings as cfg
from aqcast.storage import chunks, month_end, month_starts
from aqcast.web import RateLimiter, _safe
from datetime import date


def test_city_config_is_valid():
    s = cfg.load()
    assert len(s.cities) == 36
    assert s.city("delhi").country == "IN"
    assert {c.fire_region for c in s.cities} <= set(s.fire_regions)


def test_city_outside_its_fire_box_is_rejected():
    city = cfg.City("Nowhere", "nowhere", "XX", 10.0, 10.0, "box")
    with pytest.raises(ValueError, match="outside"):
        cfg.check((city,), {"box": (20.0, 20.0, 30.0, 30.0)})


def test_rate_limiter_spaces_requests():
    t = [0.0]
    waits = []

    def sleep(d):
        waits.append(d)
        t[0] += d

    lim = RateLimiter(per_minute=2, gap=1.0, clock=lambda: t[0], sleep=sleep)
    for _ in range(3):
        lim.wait()
    # 2nd request waits for the 1 s gap; 3rd waits until the first leaves the 60 s window
    assert waits[0] == pytest.approx(1.0)
    assert sum(waits) >= 60.0


def test_rate_limiter_counts_weight_per_day():
    t = [0.0]
    lim = RateLimiter(per_day=10, clock=lambda: t[0], sleep=lambda d: t.__setitem__(0, t[0] + d))
    lim.wait(6)
    lim.wait(4)
    lim.wait(3)                       # would make 13 > 10 in a day: waits until the first request is a day old
    assert t[0] >= 86400


def test_key_in_url_is_hidden():
    url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/0123456789abcdef0123456789abcdef/VIIRS_SNPP_NRT/1,2,3,4/1"
    assert "0123456789abcdef" not in _safe(url)
    assert "VIIRS_SNPP_NRT" in _safe(url)


def test_date_helpers():
    assert month_starts(date(2022, 11, 15), date(2023, 2, 1)) == [date(2022, 11, 1), date(2022, 12, 1),
                                                                  date(2023, 1, 1), date(2023, 2, 1)]
    assert month_end(date(2024, 2, 1)) == date(2024, 2, 29)
    assert chunks(date(2024, 1, 1), date(2024, 1, 25), 10) == [
        (date(2024, 1, 1), date(2024, 1, 10)), (date(2024, 1, 11), date(2024, 1, 20)),
        (date(2024, 1, 21), date(2024, 1, 25))]
