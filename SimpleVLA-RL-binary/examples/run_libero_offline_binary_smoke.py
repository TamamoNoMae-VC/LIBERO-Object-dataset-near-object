import argparse
import importlib.util
import json
import sys
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "verl" / "utils" / "dataset" / "libero_offline_dataset.py"
SPEC = importlib.util.spec_from_file_location("libero_offline_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
LiberoOfflineEpisodeDataset = MODULE.LiberoOfflineEpisodeDataset
collate_offline_episode_batch = MODULE.collate_offline_episode_batch


def main():
    parser = argparse.ArgumentParser(description="Smoke test the binary LIBERO offline dataset.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-num", type=int, default=4)
    args = parser.parse_args()

    dataset = LiberoOfflineEpisodeDataset(
        manifest_path=args.manifest,
        reward_mode="binary",
        sample_num=args.sample_num,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_offline_episode_batch)
    batch = next(iter(loader))
    summary = {
        "dataset_len": len(dataset),
        "batch_size": int(batch["episode_id"].shape[0]),
        "reward_mode": str(batch["reward_mode"][0]),
        "step_reward_shape": list(batch["step_reward"].shape),
        "reward_components_shape": list(batch["reward_components"].shape),
        "trajectory_length": batch["trajectory_length"].tolist(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
