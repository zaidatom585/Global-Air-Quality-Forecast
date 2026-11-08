# Global Air Quality Forecast

[![ci](https://github.com/zaidatom585/Global-Air-Quality-Forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/zaidatom585/Global-Air-Quality-Forecast/actions/workflows/ci.yml)

Forecasts PM2.5 24–72 hours ahead for 36 cities on six continents, using satellite fire detections, ground sensors
and weather, and measures how much it improves on the official CAMS forecast. It also asks which driver dominates
each city's pollution: upwind fires, coal, dust or the weather.

The same container images run in Docker Compose, in Kubernetes on Docker Desktop, and in GitHub Actions.

**Results at a glance**

- Data: 2.4 million hourly PM2.5 readings from 187 sensors, 94 million satellite fire detections, and weather
  and CAMS history for 36 cities since August 2022.
- Model: errors **27% smaller than persistence** on held-out months (17% on the latest 90 days). In
  fire-prone cities, wildfire smoke explains up to 17% of the worst forecasts (Edmonton), against 8% overall.
- Live: the first 266 real forecasts had errors **29% smaller than persistence and 57% smaller than CAMS**
  (31 Oct – 8 Nov 2026; early figures that grow daily).
- Running: Kubernetes refreshes the data and forecasts every 6 hours; a FastAPI service serves them; a Power BI
  report shows them; CI tests, scans and publishes every image.

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
docker compose run --rm tests   # 61 tests, no network needed

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

## Power BI report

A four-page Power BI report built on the forecast files and the API
([`powerbi/Global_Air_Quality_Forecast.pbix`](powerbi/Global_Air_Quality_Forecast.pbix),
[PDF](powerbi/Global_Air_Quality_Forecast.pdf)):

| Page | Shows |
| --- | --- |
| Overview | Every city's forecast for the next 3 days, on a map and in a table coloured by US EPA category |
| City detail | The past week of measured PM2.5 running into the forecast, next to CAMS and persistence, with the drivers behind it |
| Model performance | Skill against persistence by city and horizon, and how much fires drive each city's worst days |
| Forecast vs actual | Live forecasts scored once their target time has passed, against CAMS and persistence |

![Overview](images/dashboard_overview.png)
![City detail](images/dashboard_city_detail.png)
![Model performance](images/dashboard_model_performance.png)
![Forecast vs actual](images/dashboard_forecast_vs_actual.png)

Data comes from `data/forecasts/model` (every forecast), `data/clean/pm25_hourly.parquet` (measured
PM2.5), `models/report` (test scores and drivers) and the API's `/cities` endpoint. To refresh, run
`sync update`, `forecast` and `prepare clean` (or let the Kubernetes pipeline do it), then refresh the report.

First live results (266 forecasts, 31 Oct – 8 Nov 2026): errors 29% smaller than persistence and 57% smaller
than CAMS. Early figures, updated as forecasts accumulate.

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
Feature effects (`top_features`, as % up or down) are SHAP values measured from the model's average forecast
across all cities, so for a polluted city its own recent PM2.5 and its location show up as large upward
effects.
Forecasts are indicative, not an official air-quality service.

## Run on Kubernetes

The same images run on the Kubernetes cluster built into Docker Desktop, which then keeps everything up to date
on its own: a CronJob refreshes the data and the forecasts every 6 hours, and the API runs as a two-copy
Deployment. The cluster uses the same `data/` and `models/` folders as Compose, so nothing is downloaded twice.

| Object | What it does |
| --- | --- |
| `Deployment/api` + `Service/api` | two copies of the API behind one address, http://localhost:8080; health checks restart a copy that stops answering, and updates roll out one copy at a time |
| `CronJob/pipeline` | at 00:20, 06:20, 12:20 and 18:20 UTC: `sync update`, `prepare clean`, then `forecast`. A failing data source is logged and the run carries on; only a failed forecast fails the job. Never two runs at once; a run missed while the PC slept starts within the hour |
| `CronJob/retrain` | rebuilds the training table and retrains; suspended, started by hand |
| `PersistentVolume`s + claims | the project's `data/` and `models/` folders; never deleted with the cluster objects |
| `Secret/aqcast-keys` | the OpenAQ and FIRMS keys, created from `.env`; not in the repository |

Every container runs as a non-root user with a read-only root file system, no Linux capabilities and no
access to the Kubernetes API, within CPU and memory limits; the API mounts the data read-only.

Setup (Docker Desktop: Settings → Kubernetes → Enable, with the kubeadm cluster; 6 GB of memory or more):

```bash
docker compose build sync train api                         # the images the cluster runs
kubectl apply -k k8s                                        # namespace, storage, API, CronJobs
kubectl -n aqcast create secret generic aqcast-keys --from-env-file=.env
kubectl -n aqcast get pods                                  # api-… pods Running and Ready
kubectl -n aqcast create job --from=cronjob/pipeline pipeline-now   # run the pipeline once, now
kubectl -n aqcast logs -f job/pipeline-now --all-containers         # follow it
```

The project folder is assumed to be `D:\Projects\smoke-forecast`; for another location, change the two paths
in `k8s/base/storage.yaml`. After changing the code, rebuild the images and run
`kubectl -n aqcast rollout restart deployment/api`; the CronJobs pick up new images on their next run.
`kubectl delete -k k8s` removes everything except the data.

To run the images CI publishes instead of local builds: `kubectl apply -k k8s/overlays/ghcr` (the packages must
be public, or the cluster needs an image pull secret).

## CI and published images

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on GitHub's machines:

| Job | What it does |
| --- | --- |
| `tests` | builds the test image and runs the whole suite |
| `manifests` | renders the Kubernetes manifests (local and registry versions) and checks them against the API schemas with kubeconform |
| `images` | builds the `sync`, `train` and `api` images, prints a Trivy report of HIGH and CRITICAL vulnerabilities, fails on any fixable CRITICAL one, saves a CycloneDX software bill of materials, and on `main` and version tags publishes the images to `ghcr.io/zaidatom585/global-air-quality-forecast/<image>` |

It also runs every Monday to rescan for vulnerabilities found since the last build, and Dependabot proposes
updates for the actions, Python packages and base image. Images are published only after the tests and
manifest checks pass, never from pull requests, and only the publishing job may write packages.

Only GitHub's and Docker's own actions are used. Trivy runs from its container image pinned to 0.69.3, the last
release before the March 2026 Trivy supply-chain compromise (CVE-2026-33634), in which most
`aquasecurity/trivy-action` tags were repointed to credential-stealing code.

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
k8s/               Kubernetes (kubectl apply -k k8s): manifests in base/; overlays/ghcr runs the published images
.github/           CI workflow and Dependabot settings
powerbi/           Power BI report and its PDF
tests/             tests with fake API responses
tools/             check_coverage.py, used to pick the cities
```
