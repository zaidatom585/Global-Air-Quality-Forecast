# Global Air Quality Forecast

Forecasts PM2.5 24–72 hours ahead for 36 cities on six continents, using satellite fire detections, ground sensors
and weather, and measures how much it improves on the official CAMS forecast. It also asks which driver dominates
each city's pollution: upwind fires, coal, dust or the weather.

The same container images run in Docker Compose, in Kubernetes on Docker Desktop, and in GitHub Actions.

**Status:** milestone 1 of 7, data collection. The model, prediction API, dashboard and Kubernetes setup follow.

## Cities

Chosen with `tools/check_coverage.py`, which checked ground-sensor coverage in 165 cities (29 Oct 2026). Each
city kept has PM2.5 data for at least 84% of the past year; together they run from the most polluted places
(Delhi, Lahore, Bakersfield) to the cleanest (Reykjavik, Seattle, Canberra). The list is in
[`config/cities.yaml`](config/cities.yaml).

## Data sources

| Source | What it gives | Key |
| --- | --- | --- |
| [OpenAQ](https://openaq.org) | Hourly PM2.5 from ground sensors: official monitors and low-cost sensors | Free API key |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov) | Satellite fire detections (VIIRS on Suomi NPP and NOAA-20) with fire radiative power | Free MAP_KEY |
| [Open-Meteo](https://open-meteo.com) | Hourly weather: reanalysis history and forecasts | None |
| [CAMS](https://atmosphere.copernicus.eu) via Open-Meteo | The official global PM2.5, dust and aerosol forecast | None |

Weather data by Open-Meteo.com (CC BY 4.0). Air-quality forecasts from the Copernicus Atmosphere Monitoring
Service (CAMS). Fire data from NASA FIRMS. Ground measurements aggregated by OpenAQ from many providers.
Downloaded data is not stored in this repository.

## Run the collector

Needs Docker Desktop.

```bash
cp .env.example .env            # then put your own keys in .env (never commit it)
docker compose build
docker compose run --rm tests   # 18 tests, no network needed

docker compose run --rm sync sensors                 # choose 3 stations per city, with all their sensor records
docker compose run --rm sync backfill --source all   # history since Aug 2022; safe to stop and rerun
docker compose run --rm sync update                  # recent data + current forecasts (meant to run hourly)
docker compose run --rm sync status                  # files, rows and newest timestamp per source
```

`backfill` takes `--source openaq|firms|weather|cams|all`, `--start`, `--end` and `--cities delhi,lahore`.
A source whose key is missing is skipped, so the others still run.

## How the data is stored

Parquet files under `data/`, one file per natural period, written atomically:

```
data/openaq/sensors.parquet                                   chosen stations and sensor records per city
data/openaq/hours/sensor=<id>/<YYYY-MM>.parquet               hourly PM2.5 per sensor
data/firms/<region>/<satellite>/<window>_<source>_<part>.parquet   fire detections, 10-day windows
data/weather/history/city=<slug>/<YYYY-MM>.parquet            hourly weather
data/cams/history/city=<slug>/<YYYY-MM>.parquet               hourly CAMS PM2.5, dust, aerosols
data/forecasts/{weather,cams}/issued=<YYYY-MM-DDTHH>/<slug>.parquet   forecasts as issued, every 6 hours
data/runs/<time>_<command>.json                               what each run did
```

Design choices:

- **Forecasts are saved as they are issued.** Open-Meteo's CAMS history is stitched from the first hours of each
  forecast run, so it is not what CAMS predicted a day or three earlier. A fair comparison with the CAMS forecast
  needs the real forecasts, so `update` saves them from day one.
- **Resumable downloads.** A period is skipped once it was downloaded after it ended; anything still changing
  (the current month, recent fire windows) is fetched again. A file always holds its whole period, so a short
  update never overwrites a full file with fewer days.
- **Fire archive vs near real time.** FIRMS's final archive trails real time by a few months. Windows past its end
  come from the near-real-time feed and are replaced once the archive catches up.
- **Polite to free APIs.** Each source has its own rate limiter and retries throttling and server errors with
  backoff. OpenAQ stays under 60 requests a minute and 2,000 an hour. Open-Meteo counts a request covering more
  than two weeks as several calls, so each request passes that weight to the limiter, which also keeps a daily
  budget shared through a file under `data/state/`, so backfills in parallel windows can't overrun it together.
  The hourly update only refreshes the current month when its copy is stale (CAMS every 6 hours, weather
  daily), which uses under 1,000 of the 10,000 calls a day.
- **One station, many records.** OpenAQ often lists a station several times (old and re-registered records,
  old and new sensor ids). Records within 500 m are grouped into one site and every sensor record is
  downloaded over its own active period, so the full 2022-2026 history is kept.

## Project layout

```
aqcast/            Python package
  sources/         openaq.py, firms.py, openmeteo.py: one module per data source
  sync.py          command line: sensors, backfill, update, snapshot, status
  web.py           rate limiting and retries
  storage.py       Parquet writes and date helpers
  settings.py      paths, keys, city list
config/            cities and fire-search regions
containers/        Dockerfiles
tests/             tests with fake API responses
tools/             check_coverage.py, used to pick the cities
```
