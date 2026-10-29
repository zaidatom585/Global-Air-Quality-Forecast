"""Paths, keys and the city list, read once from the environment and config/cities.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(os.environ.get("AQ_CITIES_FILE", ROOT / "config" / "cities.yaml"))
DATA_DIR = Path(os.environ.get("AQ_DATA_DIR", ROOT / "data"))

# The global CAMS forecast history on Open-Meteo starts in August 2022, so training data starts there too.
HISTORY_START = date(2022, 8, 1)


@dataclass(frozen=True)
class City:
    name: str
    slug: str
    country: str
    lat: float
    lon: float
    fire_region: str
    driver: str = ""


@dataclass(frozen=True)
class Settings:
    cities: tuple[City, ...]
    fire_regions: dict[str, tuple[float, float, float, float]]
    data_dir: Path = DATA_DIR
    openaq_key: str | None = field(default=None, repr=False)
    firms_key: str | None = field(default=None, repr=False)

    def city(self, slug: str) -> City:
        for c in self.cities:
            if c.slug == slug:
                return c
        raise KeyError(f"Unknown city '{slug}'. Known: {', '.join(c.slug for c in self.cities)}")

    def pick(self, slugs: list[str] | None) -> list[City]:
        return list(self.cities) if not slugs else [self.city(s) for s in slugs]


def load(config_file: Path = CONFIG_FILE, data_dir: Path = DATA_DIR) -> Settings:
    raw = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    cities = tuple(City(**c) for c in raw["cities"])
    regions = {k: tuple(float(x) for x in v) for k, v in raw["fire_regions"].items()}
    check(cities, regions)
    return Settings(cities=cities, fire_regions=regions, data_dir=Path(data_dir),
                    openaq_key=os.environ.get("OPENAQ_API_KEY") or None,
                    firms_key=os.environ.get("FIRMS_MAP_KEY") or None)


def check(cities: tuple[City, ...], regions: dict[str, tuple[float, ...]]) -> None:
    """Fail early on a config mistake: duplicate slugs, unknown regions, or a city outside its fire box."""
    slugs = [c.slug for c in cities]
    dupes = {s for s in slugs if slugs.count(s) > 1}
    if dupes:
        raise ValueError(f"Duplicate city slugs: {sorted(dupes)}")
    for name, (w, s, e, n) in regions.items():
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise ValueError(f"Fire region {name} is not a valid [west, south, east, north] box")
    for c in cities:
        if c.fire_region not in regions:
            raise ValueError(f"{c.name}: unknown fire_region '{c.fire_region}'")
        w, s, e, n = regions[c.fire_region]
        if not (w < c.lon < e and s < c.lat < n):
            raise ValueError(f"{c.name} ({c.lat}, {c.lon}) lies outside its fire region {c.fire_region}")
