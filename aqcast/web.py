"""HTTP access with rate limits and retries, shared by every data source."""
from __future__ import annotations

import logging
import time
from collections import deque

import requests

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class RateLimiter:
    """Blocks until a request fits inside every configured window.

    per_minute / per_hour / per_day: maximum request weight in any rolling 60 s / 1 h / 24 h window.
    A plain request weighs 1; Open-Meteo counts a long or wide request as several calls, so it passes a weight.
    gap: minimum seconds between two requests.
    """

    def __init__(self, per_minute: float | None = None, per_hour: float | None = None, per_day: float | None = None,
                 gap: float = 0.0, clock=time.monotonic, sleep=time.sleep):
        self.windows = [(w, n) for w, n in ((60.0, per_minute), (3600.0, per_hour), (86400.0, per_day)) if n]
        self.gap = gap
        self.clock, self.sleep = clock, sleep
        self.sent: deque[tuple[float, float]] = deque()      # (time, weight)

    def wait(self, weight: float = 1.0) -> None:
        while True:
            now = self.clock()
            longest = max((w for w, _ in self.windows), default=0.0)
            while self.sent and now - self.sent[0][0] > longest:
                self.sent.popleft()
            delay = self.gap - (now - self.sent[-1][0]) if self.sent else 0.0
            for window, limit in self.windows:
                recent = [(t, w) for t, w in self.sent if now - t < window]
                used = sum(w for _, w in recent)
                if recent and used + weight > limit:
                    # wait until enough of the oldest requests leave the window
                    freed, until = 0.0, recent[0][0]
                    for t, w in recent:
                        freed += w
                        until = t
                        if used - freed + weight <= limit:
                            break
                    delay = max(delay, window - (now - until) + 0.05)
            if delay <= 0:
                self.sent.append((now, weight))
                return
            if delay > 30:
                log.info("Rate limit reached; waiting %.0f s", delay)
            self.sleep(delay)


class Fetcher:
    """GET with a rate limiter, retries on throttling and server errors, and readable failures."""

    def __init__(self, name: str, limiter: RateLimiter, headers: dict | None = None, timeout: float = 120,
                 retries: int = 5, session: requests.Session | None = None, sleep=time.sleep):
        self.name, self.limiter, self.timeout, self.retries = name, limiter, timeout, retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "Global-Air-Quality-Forecast (github.com/zaidatom585)"})
        if headers:
            self.session.headers.update(headers)
        self.sleep = sleep
        self.requests_made = 0

    def get(self, url: str, params: dict | None = None, weight: float = 1.0) -> requests.Response:
        last = None
        for attempt in range(1, self.retries + 1):
            self.limiter.wait(weight)
            self.requests_made += 1
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = f"{type(e).__name__}"
                self._back_off(attempt, None, last)
                continue
            if r.status_code in RETRY_STATUS:
                last = f"HTTP {r.status_code}"
                self._back_off(attempt, r.headers.get("Retry-After"), last)
                continue
            if r.status_code >= 400:
                raise SourceError(f"{self.name}: HTTP {r.status_code} for {_safe(url)}: {r.text[:300]}")
            return r
        raise SourceError(f"{self.name}: gave up on {_safe(url)} after {self.retries} attempts ({last})")

    def _back_off(self, attempt: int, retry_after: str | None, why: str) -> None:
        wait = float(retry_after) if retry_after and retry_after.isdigit() else min(15 * 2 ** (attempt - 1), 600)
        log.warning("%s: %s, retry %d/%d in %.0f s", self.name, why, attempt, self.retries, wait)
        self.sleep(wait)


class SourceError(RuntimeError):
    pass


class MissingKey(SourceError):
    """A source's API key is not set: the source is skipped, which is not counted as a failure."""


def _safe(url: str) -> str:
    """Hide a key embedded in a URL path (FIRMS puts its MAP_KEY there) before it reaches a log."""
    parts = url.split("/")
    return "/".join("***" if len(p) >= 24 and p.isalnum() else p for p in parts)
