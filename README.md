# Global Air Quality Forecast

Forecasts PM2.5 24–72 hours ahead for 36 cities on six continents, using satellite fire detections, ground sensors
and weather, and measures how much it improves on the official CAMS forecast. It also asks which driver dominates
each city's pollution: upwind fires, coal, dust or the weather.

The same container images run in Docker Compose, in Kubernetes on Docker Desktop, and in GitHub Actions.

**Status:** milestone 4 of 7, the forecast API. Data collection (1), data checks and the training table (2)
and the model (3) are done; the dashboard and Kubernetes setup follow.

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
docker compose run --rm tests   # 46 tests, no network needed

docker compose run --rm sync sensors                 # choose 3 stations per city, with all their sensor records
docker compose run --rm sync backfill --source all   # history since Aug 2022; safe to stop and rerun
docker compose run --rm sync update                  # recent data + current forecasts (meant to run hourly)
docker compose run --rm sync status                  # files, rows and newest timestamp per source
```

`backfill` takes `--source openaq|firms|weather|cams|all`, `--start`, `--end` and `--cities delhi,lahore`.
A source whose key is missing is skipped, so the others still run.

## Prepare the training data

```bash
docker compose run --rm prepare all        # or one step: clean | fires | features
```

1. **clean**: PM2.5 checks. Hour stamps are moved to whole UTC hours first (stations in India and Nepal report
   hours starting at :30 or :15 UTC). Indoor, test and calibration units are dropped by name, and others listed with a
   reason in [`config/sensor_exclusions.yaml`](config/sensor_exclusions.yaml). Each hourly reading is flagged
   if it is missing, covers under 75% of the hour (unless the sensor's coverage figure is low nearly all the
   time, which means the figure is wrong), is below -5 or above 1,500 µg/m³, repeats the same value
   for 24 hours (a stuck sensor), or is a one-off spike far above the sensor's running median. A station's
   value is the median of its sensor records and the city's value the median of its stations. A quality
   report per city lists coverage and the share of readings each rule removed.
2. **fires**: gas flares, steel works and other permanent hotspots (fire detections in the same 2 km cell in
   at least 35% of months, spread over 9 or more calendar months) are removed, then fire power within 1,000 km of each city is summed per 6 hours
   by distance ring (0-100, 100-300, 300-600, 600-1,000 km) and compass direction.
3. **features**: one row per city, forecast time (every 6 hours) and horizon (24, 48, 72 h). The target is
   the mean PM2.5 over the six hours ending at the horizon. Features: recent PM2.5, CAMS at forecast time,
   weather now and at the target time, fire power in the last 24 and 72 hours per ring, and fire power
   upwind (each direction weighted by how directly the wind blows from it towards the city).

Notes on the data, from the quality report: OpenAQ has no records for the Delhi, Kolkata and Patna stations
between November 2022 and January 2025, so those cities have about two years of data rather than four.

No feature uses data from after the forecast time, with one stated exception: the weather at the target
time comes from reanalysis (what happened), standing in for the weather forecast the model gets in real use.
Scores on this table are therefore a little optimistic; the saved forecast snapshots allow the honest test.
CAMS at the target time is left out for the same reason, because its history is built from analyses.

## Train and test the model

```bash
docker compose run --rm train            # about 10-20 minutes; --quick for a fast check
```

One LightGBM model for all cities and horizons, trained on log(1 + PM2.5) so that cities from 5 to 120 µg/m³
count alike. It is tested two ways, always on rows it never saw:

- **held-out months**: months are split into 5 groups; each group is predicted by a model trained on the other
  four, with 4 days left out on each side so no training row overlaps a test row's 72-hour window;
- **recent**: the last 90 days, predicted by a model trained only on earlier data.

Baselines on the same rows: **persistence** (the next days look like the last 24 hours) and the **CAMS**
forecast where a saved forecast exists (from the start of collection, so this comparison grows over time).
A log-scale model predicts the typical value, which sits below the average, so it under-shoots smog peaks; a
per-city scale factor learned on other test rows corrects that, and both versions are scored. SHAP values
(LightGBM's built-in TreeSHAP) are grouped into recent PM2.5, weather, fires, CAMS, time and place to show what
drives each city's forecast, over all forecasts and over each city's worst 10%, where smoke episodes are.
Results go to `models/report/` (`summary.md` has the tables).

## Live forecasts and the API

```bash
docker compose run --rm sync update      # latest sensor readings, fires and weather/CAMS forecasts
docker compose run --rm forecast         # forecasts for every city from that data
docker compose up -d api                 # http://localhost:8000/docs
```

The forecast job builds the same features as in training at each city's latest 6-hour issue time. PM2.5
comes from the last 10 days of readings, through the same checks. Weather is stitched from reanalysis, the
first hours of each saved forecast and the latest forecast for the hours ahead; fires from the last 15 days of
detections, without the permanent hotspots. It saves every forecast (for scoring later) and a `latest.json`
that the API serves.

| Endpoint | Returns |
| --- | --- |
| `GET /health` | whether a forecast exists and how old it is |
| `GET /cities` | the cities with their latest observed PM2.5 |
| `GET /forecast` | the latest 24, 48 and 72-hour forecast for every city |
| `GET /forecast/{city}` | one city in full: model and bias-corrected values, an indicative US AQI category, persistence and CAMS for comparison, the share of each driver, and the features that moved the forecast most (as % up or down) |
| `GET /model` | how the model was trained and how it scored, overall and per city |

The API only reads files, so it starts in a second, needs no model library and can run as several copies.
Forecasts are indicative, not an official air-quality service.

## How the data is stored

Parquet files under `data/`, one file per natural period, written atomically:

```
data/openaq/sensors.parquet                                   chosen stations and sensor records per city
data/openaq/hours/sensor=<id>/<YYYY-MM>.parquet               hourly PM2.5 per sensor
data/firms/<region>/<satellite>/<window>_<source>_<part>.parquet   fire detections, 5-day windows
data/weather/history/city=<slug>/<YYYY-MM>.parquet            hourly weather
data/cams/history/city=<slug>/<YYYY-MM>.parquet               hourly CAMS PM2.5, dust, aerosols
data/forecasts/{weather,cams}/issued=<YYYY-MM-DDTHH>/<slug>.parquet   forecasts as issued, every 6 hours
data/runs/<time>_<command>.json                               what each run did
data/clean/pm25_hourly.parquet                                checked hourly PM2.5 per city
data/clean/quality_report.json, sensor_quality.parquet        what the checks removed, per city and sensor
data/clean/fire_static_cells.parquet, fires_by_city.parquet   permanent hotspots; fire power around each city
data/features/training.parquet                                the model's training table
data/forecasts/model/issued=<YYYY-MM-DDTHH>/forecast.parquet  the model's live forecasts, kept for scoring
data/forecasts/model/latest.json                              what the API serves
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
  clean.py         PM2.5 checks and the hourly city series
  fires.py         permanent hotspots; fire power by distance and direction
  features.py      the training table
  prepare.py       command line: clean, fires, features, all
  model.py         training, tests against baselines, SHAP drivers
  train.py         command line: train
  forecast.py      live forecasts from the latest data (command line too)
  api.py           FastAPI app serving the forecasts
  web.py           rate limiting and retries
  storage.py       Parquet writes and date helpers
  settings.py      paths, keys, city list
config/            cities, fire-search regions, excluded sensors
containers/        Dockerfiles
tests/             tests with fake API responses
tools/             check_coverage.py, used to pick the cities
```
