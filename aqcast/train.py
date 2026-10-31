"""Command line for training and testing the forecast model.

    python -m aqcast.train            train, test against persistence and CAMS, explain, save to models/
    python -m aqcast.train --quick    the same with a small model, to check everything runs (a minute or two)

Needs data/features/training.parquet (python -m aqcast.prepare all). Writes models/ (see aqcast/model.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from aqcast import model
from aqcast import settings as cfg


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(prog="python -m aqcast.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="small model, for a fast check")
    args = p.parse_args(argv)
    t0 = time.monotonic()
    try:
        result = model.run(cfg.load(), quick=args.quick)
    except FileNotFoundError as e:
        logging.error("%s. Run first:  docker compose run --rm prepare all", e)
        return 1
    result["seconds"] = round(time.monotonic() - t0, 1)
    logging.info("train: %s", json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
