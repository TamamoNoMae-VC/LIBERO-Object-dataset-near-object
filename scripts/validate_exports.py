from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from libero_a2b.config import load_config
from libero_a2b.storage import write_json
from libero_a2b.validators import validate_reward_exports


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate binary and dense exports against raw data.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "libero_a2b_v1.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    report = validate_reward_exports(cfg)
    write_json(cfg.resolve_path(cfg.paths.reports_dir) / "export_validation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
