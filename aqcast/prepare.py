"""Command line for turning downloaded data into the model's training table.

    python -m aqcast.prepare clean       PM2.5 checks and the hourly city series, with a quality report
    python -m aqcast.prepare fires       permanent hotspots removed; fire power around each city
    python -m aqcast.prepare features    the training table (needs clean; uses fires if present)
    python -m aqcast.prepare all         all three, in order

Reads data/ as written by aqcast.sync and writes data/clean/ and data/features/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from aqcast import clean, features, fires
from aqcast import settings as cfg
from aqcast.storage import utc_now

log = logging.getLogger("aqcast")
STEPS = {"clean": clean.run, "fires": fires.run, "features": features.run}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(prog="python -m aqcast.prepare", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=(*STEPS, "all"))
    args = p.parse_args(argv)
    s = cfg.load()
    steps = list(STEPS) if args.step == "all" else [args.step]
    results, started, failed = {}, utc_now(), []
    for name in steps:
        t0 = time.monotonic()
        log.info("--- %s", name)
        try:
            results[name] = {"ok": True, **STEPS[name](s)}
        except (RuntimeError, FileNotFoundError) as e:
            log.error("%s: %s", name, e)
            results[name] = {"ok": False, "error": str(e)}
            failed.append(name)
        results[name]["seconds"] = round(time.monotonic() - t0, 1)
        log.info("%s: %s", name, json.dumps(results[name], default=str))
        if failed:
            break                                    # later steps need this one's output
    runs = Path(s.data_dir) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{started:%Y%m%dT%H%M%S}_prepare_{args.step}.json").write_text(
        json.dumps({"command": f"prepare {args.step}", "results": results}, indent=2, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
