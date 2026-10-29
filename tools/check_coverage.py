"""Check how well OpenAQ's ground sensors cover candidate cities for the wildfire smoke project.

For each city it finds every PM2.5 sensor within 25 km, then measures how complete the daily record is
over the last year for the best three sensors. Results print as a table and are saved to coverage_report.csv.

Needs a free OpenAQ API key in the environment variable OPENAQ_API_KEY (never put the key in this file).

    python check_coverage.py              # all cities below
    python check_coverage.py Athens Sydney
"""
import csv
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import requests

API = "https://api.openaq.org/v3"
PM25 = 2                      # OpenAQ parameter id for PM2.5
RADIUS_M = 25_000             # the API's maximum search radius
TOP_SENSORS = 3
PAUSE_S = 1.1                 # stay under the free tier's 60 requests a minute

# Cities that regularly get wildfire or agricultural-burning smoke, spread across continents.
CITIES = {
    "Los Angeles": (34.05, -118.24),
    "Sacramento": (38.58, -121.49),
    "Portland": (45.52, -122.68),
    "Seattle": (47.61, -122.33),
    "Vancouver": (49.28, -123.12),
    "Calgary": (51.05, -114.07),
    "Edmonton": (53.55, -113.49),
    "Denver": (39.74, -104.99),
    "Athens": (37.98, 23.73),
    "Lisbon": (38.72, -9.14),
    "Madrid": (40.42, -3.70),
    "Sydney": (-33.87, 151.21),
    "Melbourne": (-37.81, 144.96),
    "Santiago": (-33.45, -70.67),
    "Sao Paulo": (-23.55, -46.63),
    "Manaus": (-3.12, -60.02),
    "Cape Town": (-33.92, 18.42),
    "Chiang Mai": (18.79, 98.98),
    "Jakarta": (-6.21, 106.85),
    # Equatorial Asia: Indonesian peat and forest fires, haze across Singapore and Malaysia
    "Singapore": (1.35, 103.82),
    "Kuala Lumpur": (3.14, 101.69),
    "Pekanbaru": (0.51, 101.45),
    "Palembang": (-2.99, 104.76),
    "Pontianak": (-0.03, 109.33),
    "Palangka Raya": (-2.21, 113.92),
    # Mainland South-East Asia: dry-season crop and forest burning (Feb-Apr)
    "Chiang Rai": (19.91, 99.83),
    "Bangkok": (13.76, 100.50),
    "Hanoi": (21.03, 105.85),
    "Vientiane": (17.97, 102.63),
    "Yangon": (16.84, 96.17),
    # South Asia: post-harvest crop burning (Oct-Nov) and spring burning
    "Delhi": (28.61, 77.21),
    "Lahore": (31.55, 74.34),
    "Amritsar": (31.63, 74.87),
    "Kathmandu": (27.72, 85.32),
    "Dhaka": (23.81, 90.41),
    # Siberia / Russian Far East: summer forest fires
    "Yakutsk": (62.03, 129.73),
    # Africa: savanna and agricultural burning
    "Kinshasa": (-4.44, 15.27),
    "Abuja": (9.08, 7.40),
    "Kampala": (0.35, 32.58),
    "Nairobi": (-1.29, 36.82),
    "Lusaka": (-15.39, 28.32),
    # South Asia, more cities: India (incl. coal belt), Pakistan, Bangladesh
    "Kolkata": (22.57, 88.36),
    "Mumbai": (19.08, 72.88),
    "Chennai": (13.08, 80.27),
    "Bengaluru": (12.97, 77.59),
    "Hyderabad": (17.39, 78.49),
    "Ahmedabad": (23.02, 72.57),
    "Jaipur": (26.91, 75.79),
    "Lucknow": (26.85, 80.95),
    "Kanpur": (26.45, 80.33),
    "Patna": (25.59, 85.14),
    "Varanasi": (25.32, 82.97),
    "Ludhiana": (30.90, 75.85),
    "Chandigarh": (30.73, 76.78),
    "Singrauli": (24.20, 82.67),
    "Korba": (22.35, 82.68),
    "Dhanbad": (23.80, 86.43),
    "Karachi": (24.86, 67.01),
    "Islamabad": (33.68, 73.05),
    "Faisalabad": (31.42, 73.09),
    "Peshawar": (34.01, 71.58),
    "Multan": (30.16, 71.52),
    "Chattogram": (22.36, 91.78),
    "Rajshahi": (24.37, 88.60),
    "Khulna": (22.85, 89.54),
    "Sylhet": (24.89, 91.87),
    # China: coal power and coal heating (northern heating season Nov-Mar), plus cleaner southern cities to compare
    "Beijing": (39.90, 116.41),
    "Tianjin": (39.13, 117.20),
    "Shijiazhuang": (38.04, 114.51),
    "Tangshan": (39.63, 118.18),
    "Taiyuan": (37.87, 112.55),
    "Datong": (40.08, 113.30),
    "Hohhot": (40.84, 111.75),
    "Ordos": (39.61, 109.78),
    "Zhengzhou": (34.75, 113.63),
    "Xi'an": (34.34, 108.94),
    "Shenyang": (41.81, 123.43),
    "Harbin": (45.80, 126.53),
    "Urumqi": (43.83, 87.62),
    "Shanghai": (31.23, 121.47),
    "Chengdu": (30.57, 104.07),
    "Guangzhou": (23.13, 113.26),
    # From IQAir's World Air Quality Report 2025 rankings and regional findings
    # India: Delhi NCR satellite cities and other top-ranked towns
    "Loni": (28.75, 77.29),
    "Ghaziabad": (28.67, 77.45),
    "Noida": (28.54, 77.39),
    "Gurugram": (28.46, 77.03),
    "Faridabad": (28.41, 77.32),
    "Meerut": (28.98, 77.71),
    "Hapur": (28.73, 77.78),
    "Muzaffarnagar": (29.47, 77.70),
    "Sri Ganganagar": (29.91, 73.88),
    "Byrnihat": (26.05, 91.87),
    # Pakistan and Bangladesh (the two most polluted countries)
    "Gujranwala": (32.16, 74.19),
    "Rawalpindi": (33.60, 73.04),
    "Sheikhupura": (31.71, 73.99),
    "Bahawalpur": (29.40, 71.68),
    "Gazipur": (24.00, 90.42),
    "Narayanganj": (23.62, 90.50),
    "Birgunj": (27.01, 84.88),
    # Central Asia and Mongolia: Tajikistan ranked third; coal heating in winter
    "Dushanbe": (38.56, 68.77),
    "Tashkent": (41.30, 69.24),
    "Bishkek": (42.87, 74.59),
    "Almaty": (43.24, 76.89),
    "Ulaanbaatar": (47.89, 106.91),
    # China: desert dust in the west, industry in the north
    "Hotan": (37.11, 79.92),
    "Kashgar": (39.47, 75.99),
    "Anyang": (36.10, 114.39),
    "Handan": (36.63, 114.54),
    "Xingtai": (37.07, 114.50),
    # Africa: Chad and DR Congo ranked fourth and fifth; Sahara dust and savanna burning
    "N'Djamena": (12.13, 15.06),
    "Kano": (12.00, 8.52),
    "Lagos": (6.52, 3.38),
    "Accra": (5.60, -0.19),
    "Cairo": (30.04, 31.24),
    # Middle East: desert dust
    "Baghdad": (33.31, 44.36),
    "Riyadh": (24.71, 46.68),
    "Kuwait City": (29.38, 47.98),
    "Manama": (26.23, 50.59),
    "Tehran": (35.69, 51.39),
    # Canada: most polluted country in North America in 2025 (wildfire smoke)
    "Toronto": (43.65, -79.38),
    "Montreal": (45.50, -73.57),
    "Winnipeg": (49.90, -97.14),
    "Yellowknife": (62.45, -114.37),
    # Latin America
    "Mexico City": (19.43, -99.13),
    "Lima": (-12.05, -77.04),
    "Bogota": (4.71, -74.07),
    # Largest cities (UN World Urbanization Prospects 2025) not already listed
    "Tokyo": (35.68, 139.69),
    "Manila": (14.60, 120.98),
    "Seoul": (37.57, 126.98),
    # USA: largest cities by population (Los Angeles, Seattle etc. are listed above)
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Houston": (29.76, -95.37),
    "Phoenix": (33.45, -112.07),
    "Philadelphia": (39.95, -75.17),
    "San Antonio": (29.42, -98.49),
    "San Diego": (32.72, -117.16),
    "Dallas": (32.78, -96.80),
    "Jacksonville": (30.33, -81.66),
    "Austin": (30.27, -97.74),
    "San Jose": (37.34, -121.89),
    # USA: most polluted (American Lung Association State of the Air 2026; IQAir 2025 for El Paso)
    "Bakersfield": (35.37, -119.02),
    "Fresno": (36.74, -119.79),
    "Visalia": (36.33, -119.29),
    "El Centro": (32.79, -115.56),
    "El Paso": (31.76, -106.49),
    # USA: cleanest (State of the Air 2026 clean lists; Honolulu as a remote-island comparison)
    "Salinas": (36.68, -121.66),
    "Chico": (39.73, -121.84),
    "Santa Barbara": (34.42, -119.70),
    "San Luis Obispo": (35.28, -120.66),
    "Honolulu": (21.31, -157.86),
    # New Zealand (winter wood-burner smoke, especially Christchurch) and more of Australia
    "Auckland": (-36.85, 174.76),
    "Wellington": (-41.29, 174.78),
    "Christchurch": (-43.53, 172.64),
    "Canberra": (-35.28, 149.13),
    "Brisbane": (-27.47, 153.03),
    "Perth": (-31.95, 115.86),
    "Hobart": (-42.88, 147.33),
    # Cleanest places: the 13 countries/territories meeting the WHO guideline in 2025 (IQAir), plus the cleanest city
    "Reykjavik": (64.15, -21.94),
    "Tallinn": (59.44, 24.75),
    "San Juan": (18.47, -66.11),
    "Panama City": (8.98, -79.52),
    "Bridgetown": (13.10, -59.61),
    "Noumea": (-22.28, 166.46),
    "Papeete": (-17.54, -149.57),
    "Nieuwoudtville": (-31.37, 19.11),
}

# Run one region at a time, e.g.  python check_coverage.py south_asia   ("asia" = south + southeast)
REGIONS = {
    "americas": ["Los Angeles", "Sacramento", "Portland", "Seattle", "Vancouver", "Calgary", "Edmonton", "Denver",
                 "Santiago", "Sao Paulo", "Manaus", "Toronto", "Montreal", "Winnipeg", "Yellowknife", "Mexico City", "Lima", "Bogota", "San Juan", "Panama City", "Bridgetown"],
    "europe": ["Athens", "Lisbon", "Madrid", "Reykjavik", "Tallinn"],
    "oceania": ["Sydney", "Melbourne", "Auckland", "Wellington", "Christchurch", "Canberra", "Brisbane", "Perth", "Hobart", "Noumea", "Papeete"],
    "southeast_asia": ["Chiang Mai", "Jakarta", "Singapore", "Kuala Lumpur", "Pekanbaru", "Palembang", "Pontianak",
                       "Palangka Raya", "Chiang Rai", "Bangkok", "Hanoi", "Vientiane", "Yangon", "Yakutsk"],
    "south_asia": ["Delhi", "Lahore", "Amritsar", "Kathmandu", "Dhaka", "Kolkata", "Mumbai", "Chennai", "Bengaluru",
                   "Hyderabad", "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Patna", "Varanasi", "Ludhiana",
                   "Chandigarh", "Singrauli", "Korba", "Dhanbad", "Karachi", "Islamabad", "Faisalabad", "Peshawar",
                   "Multan", "Chattogram", "Rajshahi", "Khulna", "Sylhet", "Loni", "Ghaziabad", "Noida", "Gurugram", "Faridabad", "Meerut", "Hapur", "Muzaffarnagar", "Sri Ganganagar", "Byrnihat", "Gujranwala", "Rawalpindi", "Sheikhupura", "Bahawalpur", "Gazipur", "Narayanganj", "Birgunj"],
    "africa": ["Cape Town", "Kinshasa", "Abuja", "Kampala", "Nairobi", "Lusaka", "N'Djamena", "Kano", "Lagos", "Accra", "Cairo", "Nieuwoudtville"],
    "central_asia": ["Dushanbe", "Tashkent", "Bishkek", "Almaty", "Ulaanbaatar"],
    "middle_east": ["Baghdad", "Riyadh", "Kuwait City", "Manama", "Tehran"],
    "ranked": ["Loni", "Delhi", "Byrnihat", "Ghaziabad", "Lahore", "Dhaka", "Kolkata", "Kinshasa", "N'Djamena",
               "Dushanbe", "Faisalabad", "Gujranwala", "Hotan", "Ulaanbaatar", "Baghdad"],
    "usa": ["New York", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas",
            "Jacksonville", "Austin", "San Jose", "Bakersfield", "Fresno", "Visalia", "El Centro", "El Paso", "Salinas",
            "Chico", "Santa Barbara", "San Luis Obispo", "Honolulu"],
    # Overlay lists (cities also belong to a region above)
    "cleanest": ["Nieuwoudtville", "Reykjavik", "Tallinn", "San Juan", "Panama City", "Bridgetown", "Noumea",
                 "Papeete", "Canberra", "Hobart", "Honolulu", "Seattle", "Wellington"],
    "us_extremes": ["Bakersfield", "Fresno", "Visalia", "El Centro", "El Paso", "Salinas", "Chico", "Santa Barbara",
                    "San Luis Obispo", "Seattle"],
    "east_asia": ["Tokyo", "Manila", "Seoul"],
    "largest": ["Jakarta", "Dhaka", "Tokyo", "Delhi", "Shanghai", "Guangzhou", "Cairo", "Manila", "Kolkata", "Seoul"],
    "china": ["Beijing", "Tianjin", "Shijiazhuang", "Tangshan", "Taiyuan", "Datong", "Hohhot", "Ordos", "Zhengzhou",
              "Xi'an", "Shenyang", "Harbin", "Urumqi", "Shanghai", "Chengdu", "Guangzhou", "Hotan", "Kashgar", "Anyang", "Handan", "Xingtai"],
}


def get(session, path, **params):
    status = None
    for attempt in range(4):
        r = session.get(f"{API}{path}", params=params, timeout=60)
        time.sleep(PAUSE_S)
        status = r.status_code
        if status == 429 or status >= 500:   # rate limited or server hiccup: back off and retry
            time.sleep(15 * (attempt + 1))
            continue
        if status == 401:
            sys.exit("OpenAQ rejected the API key (401). Check OPENAQ_API_KEY.")
        r.raise_for_status()
        return r.json().get("results", [])
    raise RuntimeError(f"gave up on {path} after repeated {status} errors")


def utc(d):
    v = (d or {}).get("utc")
    return datetime.fromisoformat(v.replace("Z", "+00:00")) if v else None


def pm25_sensors(session, lat, lon):
    """Every PM2.5 sensor within the radius, with its location's details."""
    locations = get(session, "/locations", coordinates=f"{lat},{lon}", radius=RADIUS_M,
                    parameters_id=PM25, limit=1000)
    out = []
    for loc in locations:
        for s in loc.get("sensors", []):
            if (s.get("parameter") or {}).get("id") != PM25:
                continue
            out.append({
                "sensor_id": s["id"],
                "location": loc.get("name"),
                "provider": (loc.get("provider") or {}).get("name"),
                "reference_monitor": bool(loc.get("isMonitor")),
                "first": utc(loc.get("datetimeFirst")),
                "last": utc(loc.get("datetimeLast")),
            })
    return out


def days_with_data(session, sensor_id, start, end, max_pages=10):
    """Number of distinct days in [start, end] with a daily PM2.5 value.

    The dates are checked here rather than trusted to the API's filter, and results are paged,
    so a sensor with a long history can't be counted as more than 100% of the year.
    """
    seen = set()
    for page in range(1, max_pages + 1):
        rows = get(session, f"/sensors/{sensor_id}/days", date_from=start.isoformat(), date_to=end.isoformat(),
                   datetime_from=start.isoformat(), datetime_to=end.isoformat(), limit=1000, page=page)
        for r in rows:
            when = utc((r.get("period") or {}).get("datetimeFrom"))
            if when and start <= when.date() <= end:
                seen.add(when.date())
        if len(rows) < 1000:
            break
    return len(seen)


def check_city(session, name, lat, lon, now):
    sensors = pm25_sensors(session, lat, lon)
    active = [s for s in sensors if s["last"] and s["last"] >= now - timedelta(days=7)]
    monitors = [s for s in active if s["reference_monitor"]]
    oldest = min((s["first"] for s in sensors if s["first"]), default=None)

    # Rank active sensors: reference monitors first, then the longest-running.
    ranked = sorted(active, key=lambda s: (not s["reference_monitor"], s["first"] or now))[:TOP_SENSORS]
    end = now.date() - timedelta(days=1)
    start = end - timedelta(days=364)
    coverage = []
    for s in ranked:
        try:
            n = days_with_data(session, s["sensor_id"], start, end)
        except (requests.HTTPError, RuntimeError) as e:   # one broken sensor shouldn't sink the whole city
            print(f"  skipped sensor {s['sensor_id']} ({s['location']}): {e}")
            continue
        coverage.append((s, round(100 * n / 365, 1)))

    best = max((c for _, c in coverage), default=0.0)
    if len(active) >= 3 and best >= 80:
        verdict = "good"
    elif active and best >= 50:
        verdict = "usable"
    else:
        verdict = "weak"
    return {
        "city": name,
        "pm25_sensors_total": len(sensors),
        "active_last_7_days": len(active),
        "active_reference_monitors": len(monitors),
        "oldest_record": oldest.date().isoformat() if oldest else "",
        "best_sensor_days_pct_last_year": best,
        "best_sensors": "; ".join(
            f"{s['location']} ({'monitor' if s['reference_monitor'] else s['provider']}, {c}%)" for s, c in coverage),
        "verdict": verdict,
    }


def main():
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        sys.exit("Set OPENAQ_API_KEY first (free key from https://explore.openaq.org).")
    regions = dict(REGIONS, asia=REGIONS["southeast_asia"] + REGIONS["south_asia"])
    wanted = []
    for arg in sys.argv[1:] or list(CITIES):
        wanted += regions.get(arg.lower(), [arg])
    wanted = list(dict.fromkeys(wanted))     # a city in two of the requested lists is checked once
    unknown = [c for c in wanted if c not in CITIES]
    if unknown:
        sys.exit(f"Unknown city: {', '.join(unknown)}. Known: {', '.join(CITIES)}")

    session = requests.Session()
    session.headers["X-API-Key"] = key
    now = datetime.now(timezone.utc)
    results = []
    for name in wanted:
        print(f"Checking {name} ...", flush=True)
        try:
            results.append(check_city(session, name, *CITIES[name], now))
        except requests.HTTPError as e:
            print(f"  failed: {e}")
            results.append({"city": name, "verdict": f"error: {e.response.status_code}"})

    cols = ["city", "verdict", "active_last_7_days", "active_reference_monitors", "pm25_sensors_total",
            "oldest_record", "best_sensor_days_pct_last_year", "best_sensors"]
    with open("coverage_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"\n{'City':<12} {'Verdict':<8} {'Active':>6} {'Monitors':>8} {'Since':>11} {'Best year %':>11}")
    for r in sorted(results, key=lambda r: (-r.get("best_sensor_days_pct_last_year", 0), r["city"])):
        print(f"{r['city']:<12} {r['verdict']:<8} {r.get('active_last_7_days', ''):>6} "
              f"{r.get('active_reference_monitors', ''):>8} {r.get('oldest_record', ''):>11} "
              f"{r.get('best_sensor_days_pct_last_year', ''):>11}")
    print("\nSaved coverage_report.csv (includes the best sensors for each city).")


if __name__ == "__main__":
    main()
