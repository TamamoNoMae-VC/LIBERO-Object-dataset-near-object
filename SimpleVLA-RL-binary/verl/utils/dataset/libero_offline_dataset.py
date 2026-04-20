import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def collate_offline_episode_batch(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}
    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors.setdefault(key, []).append(val)
            else:
                non_tensors.setdefault(key, []).append(val)

    output = {}
    for key, values in tensors.items():
        shapes = [tuple(val.shape) for val in values]
        if len(set(shapes)) == 1:
            output[key] = torch.stack(values, dim=0)
            continue
        padding_value = False if values[0].dtype == torch.bool else 0
        output[key] = pad_sequence(values, batch_first=True, padding_value=padding_value)
    for key, values in non_tensors.items():
        output[key] = np.array(values, dtype=object)
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _load_episode_npz(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    metadata = json.loads(str(loaded["metadata_json"].item()))
    reward = json.loads(str(loaded["reward_json"].item()))
    return {
        "metadata": metadata,
        "reward": reward,
        "actions": loaded["actions"].astype(np.float32),
        "dones": loaded["dones"].astype(bool),
        "env_rewards": loaded["env_rewards"].astype(np.float32),
        "eef_positions": loaded["eef_positions"].astype(np.float32),
        "object_positions": loaded["object_positions"].astype(np.float32),
        "target_positions": loaded["target_positions"].astype(np.float32),
        "gripper_positions": loaded["gripper_positions"].astype(np.float32),
        "agentview_images": loaded["agentview_images"].astype(np.uint8),
        "robot0_eye_in_hand_images": loaded["robot0_eye_in_hand_images"].astype(np.uint8),
    }


class LiberoOfflineEpisodeDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | None = None,
        index_path: str | None = None,
        reward_mode: str | None = None,
        include_images: bool = False,
        sample_num: int = -1,
    ):
        if manifest_path is None and index_path is None:
            raise ValueError("Provide either manifest_path or index_path.")

        if manifest_path is not None:
            with Path(manifest_path).open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.index_path = Path(manifest["index_path"])
            self.reward_mode = reward_mode or manifest.get("reward_mode", "binary")
        else:
            self.index_path = Path(index_path)
            self.reward_mode = reward_mode or "binary"

        self.include_images = include_images
        self.rows = _read_jsonl(self.index_path)
        if sample_num > 0:
            self.rows = self.rows[:sample_num]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.rows[item]
        episode = _load_episode_npz(Path(row["episode_path"]))
        reward = episode.get("reward", {})
        reward_components = np.asarray(reward.get("reward_components", []), dtype=np.float32)
        if reward_components.size == 0:
            reward_components = np.zeros((len(episode["actions"]), 0), dtype=np.float32)
        batch = {
            "episode_id": torch.tensor(int(row["episode_id"]), dtype=torch.int64),
            "trajectory_length": torch.tensor(int(row["trajectory_length"]), dtype=torch.int64),
            "actions": torch.tensor(episode["actions"], dtype=torch.float32),
            "dones": torch.tensor(episode["dones"], dtype=torch.bool),
            "step_reward": torch.tensor(np.asarray(reward.get("step_reward", []), dtype=np.float32)),
            "reward_components": torch.tensor(reward_components, dtype=torch.float32),
            "cumulative_return": torch.tensor(
                np.asarray(reward.get("cumulative_return", []), dtype=np.float32)
            ),
            "return_to_go": torch.tensor(np.asarray(reward.get("return_to_go", []), dtype=np.float32)),
            "eef_positions": torch.tensor(episode["eef_positions"], dtype=torch.float32),
            "object_positions": torch.tensor(episode["object_positions"], dtype=torch.float32),
            "target_positions": torch.tensor(episode["target_positions"], dtype=torch.float32),
            "gripper_positions": torch.tensor(episode["gripper_positions"], dtype=torch.float32),
            "payload_hash": row["payload_hash"],
            "episode_path": row["episode_path"],
            "task_name": row["task_name"],
            "reward_mode": reward.get("reward_mode", self.reward_mode),
            "metadata_json": json.dumps(episode["metadata"], ensure_ascii=False),
            "reward_component_names": reward.get("reward_component_names", []),
        }
        if self.include_images:
            batch["agentview_images"] = torch.tensor(episode["agentview_images"], dtype=torch.uint8)
            batch["robot0_eye_in_hand_images"] = torch.tensor(
                episode["robot0_eye_in_hand_images"], dtype=torch.uint8
            )
        return batch
