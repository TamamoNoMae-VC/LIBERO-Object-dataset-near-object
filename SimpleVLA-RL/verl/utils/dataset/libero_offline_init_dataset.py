import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _load_episode_metadata(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    return json.loads(str(loaded["metadata_json"].item()))


class LiberoOfflineInitDataset(Dataset):
    def __init__(self, manifest_path: str, sample_num: int = -1):
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.index_path = Path(manifest["index_path"])
        self.rows = _read_jsonl(self.index_path)
        if sample_num > 0:
            self.rows = self.rows[:sample_num]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        metadata = _load_episode_metadata(Path(row["episode_path"]))
        object_quat = metadata.get("object_pose_initial_quat") or metadata.get("object_spawn_quat") or [0.0, 0.0, 0.0, 1.0]
        return {
            "task_id": torch.tensor(int(metadata.get("task_id", row.get("task_id", 0))), dtype=torch.int64).unsqueeze(0),
            "trial_id": torch.tensor(0, dtype=torch.int64).unsqueeze(0),
            "trial_seed": torch.tensor(-1, dtype=torch.int64).unsqueeze(0),
            "task_suite_name": metadata.get("task_family", row.get("task_family", "libero_object")),
            "custom_object_position": np.asarray(metadata["object_pose_commanded_initial"], dtype=np.float32),
            "custom_object_quat": np.asarray(object_quat, dtype=np.float32),
            "custom_target_position": np.asarray(metadata["b_position"], dtype=np.float32),
            "custom_reference_position": np.asarray(metadata["reference_object_pose"], dtype=np.float32),
            "movable_object_name": metadata.get(
                "resolved_movable_body",
                metadata.get("movable_object_name", "orange_juice"),
            ),
            "reference_object_name": metadata.get("resolved_reference_body", metadata.get("reference_object_name", "basket")),
            "offline_source_episode": int(metadata.get("episode_id", row.get("episode_id", index))),
            "data_source": "offline_init",
        }
