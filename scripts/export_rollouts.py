from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from libero_a2b.config import load_config
from libero_a2b.storage import load_episode_npz, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-export rollout videos from raw successful episodes.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "libero_a2b_v1.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    rows = read_jsonl(cfg.resolve_path(cfg.paths.raw_dir) / "index.jsonl")
    rollout_dir = cfg.resolve_path(cfg.paths.rollouts_dir)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        episode = load_episode_npz(row["episode_path"])
        output_path = rollout_dir / f"episode_{row['episode_id']:06d}.mp4"
        with imageio.get_writer(output_path, fps=20) as writer:
            for frame in episode["observations"]["agentview_images"]:
                writer.append_data(frame)
    print(f"Exported {len(rows)} rollout videos to {rollout_dir}")


if __name__ == "__main__":
    main()
