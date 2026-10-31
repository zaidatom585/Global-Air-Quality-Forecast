# Images for the whole pipeline, built from one shared base. Build stages:
#   runtime (default): python -m aqcast.sync (data collection) and python -m aqcast.prepare, non-root, data in /data
#   train:             python -m aqcast.train (model training and testing), writes models to /models
#   test:              the test suite, run with `docker compose run --rm tests`
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore
WORKDIR /app
COPY requirements/sync.txt requirements/sync.txt
RUN pip install -r requirements/sync.txt
COPY config/ config/
COPY aqcast/ aqcast/

# LightGBM and the OpenMP library it needs; only the training and test images carry them.
FROM base AS ml
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements/model.txt requirements/model.txt
RUN pip install -r requirements/model.txt

FROM ml AS test
COPY requirements/dev.txt requirements/dev.txt
RUN pip install -r requirements/dev.txt
COPY pytest.ini pytest.ini
COPY tests/ tests/
CMD ["python", "-m", "pytest", "-q"]

FROM ml AS train
RUN useradd --create-home --uid 10001 aq && mkdir -p /data /models && chown aq /data /models
USER aq
ENV AQ_DATA_DIR=/data \
    AQ_MODEL_DIR=/models
ENTRYPOINT ["python", "-m", "aqcast.train"]

FROM base AS runtime
RUN useradd --create-home --uid 10001 aq && mkdir -p /data && chown aq /data
USER aq
ENV AQ_DATA_DIR=/data
ENTRYPOINT ["python", "-m", "aqcast.sync"]
CMD ["update"]
