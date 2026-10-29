# Data collector image. Build stages:
#   runtime (default): python -m aqcast.sync, running as a non-root user, data in /data
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

FROM base AS test
COPY requirements/dev.txt requirements/dev.txt
RUN pip install -r requirements/dev.txt
COPY tests/ tests/
CMD ["python", "-m", "pytest", "-q"]

FROM base AS runtime
RUN useradd --create-home --uid 10001 aq && mkdir -p /data && chown aq /data
USER aq
ENV AQ_DATA_DIR=/data
ENTRYPOINT ["python", "-m", "aqcast.sync"]
CMD ["update"]
