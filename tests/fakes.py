"""A stand-in for aqcast.web.Fetcher that answers from a function instead of the internet."""
from __future__ import annotations

import json


class FakeResponse:
    def __init__(self, payload=None, text=None, status=200):
        self._payload, self.status_code = payload, status
        self.text = text if text is not None else json.dumps(payload)
        self.headers = {}

    def json(self):
        return self._payload if self._payload is not None else json.loads(self.text)


class FakeFetcher:
    def __init__(self, answer):
        self.answer = answer          # answer(url, params) -> FakeResponse
        self.calls = []
        self.requests_made = 0

    def get(self, url, params=None, weight=1, retries=None):
        self.calls.append((url, dict(params or {})))
        self.requests_made += 1
        return self.answer(url, params or {})
