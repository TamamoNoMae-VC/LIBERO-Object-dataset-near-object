from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .storage import read_jsonl, write_json


def export_simplevla_manifest(cfg: PipelineConfig, reward_mode: str, dataset_dir: Path, output_root: Path) -> Path:
    index_rows = read_jsonl(dataset_dir / "index.jsonl")
    payload = {
        "dataset_name": f"libero_a2b_{reward_mode}",
        "reward_mode": reward_mode,
        "task_family": cfg.task.family,
        "master_seed": cfg.collection.master_seed,
        "successful_episode_count": len(index_rows),
        "dataset_root": str(dataset_dir.resolve()),
        "index_path": str((dataset_dir / "index.jsonl").resolve()),
        "fields": {
            "episode_path": "npz episode payload",
            "payload_hash": "hash over trajectory payload excluding reward annotations",
            "step_reward": "per-step rewards",
            "reward_components": "component rewards, if shaped",
            "return_to_go": "offline RTG",
        },
    }
    return write_json(output_root / "offline_dataset_manifest.json", payload)


def write_variant_readme(output_root: Path, reward_mode: str) -> Path:
    text = (
        f"# SimpleVLA-RL {reward_mode.title()} Variant\n\n"
        f"This copy is prepared to work with the LIBERO A-to-B successful-only {reward_mode} dataset.\n"
        "Use `offline_dataset_manifest.json` and the example scripts in `examples/` to inspect and load the data.\n"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    readme_path = output_root / "OFFLINE_DATASET_README.md"
    readme_path.write_text(text, encoding="utf-8")
    return readme_path
