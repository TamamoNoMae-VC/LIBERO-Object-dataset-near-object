from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from libero_a2b.collector import collect_successful_trajectories, run_preflight
from libero_a2b.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect successful-only LIBERO A-to-B raw trajectories.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "libero_a2b_v1.yaml"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.preflight_only:
        print(json.dumps(run_preflight(), indent=2))
        return

    summary = collect_successful_trajectories(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
