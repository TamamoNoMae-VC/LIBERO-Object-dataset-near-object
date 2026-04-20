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
from libero_a2b.rewards import build_binary_rewards
from libero_a2b.storage import append_jsonl, load_episode_npz, read_jsonl, save_episode_npz, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive binary rewards from the raw successful LIBERO dataset.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "libero_a2b_v1.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    raw_rows = read_jsonl(cfg.resolve_path(cfg.paths.raw_dir) / "index.jsonl")
    out_dir = cfg.resolve_path(cfg.paths.binary_dir)
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    if index_path.exists():
        index_path.unlink()

    derived_rows = []
    for row in raw_rows:
        episode = load_episode_npz(row["episode_path"])
        episode["reward"] = build_binary_rewards(episode, broadcast=cfg.export.binary_reward_broadcast)
        output_path = episodes_dir / f"episode_{row['episode_id']:06d}.npz"
        save_episode_npz(episode, output_path)
        derived_row = dict(row)
        derived_row.update({"episode_path": str(output_path.resolve()), "reward_mode": "binary"})
        derived_rows.append(derived_row)
        append_jsonl(index_path, [derived_row])

    summary = {"episode_count": len(derived_rows), "reward_mode": "binary"}
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
