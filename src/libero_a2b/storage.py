from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _to_np_object(items: list[Any]) -> np.ndarray:
    return np.array(items, dtype=object)


def save_episode_npz(episode: dict[str, Any], output_path: str | Path) -> Path:
    output_path = ensure_parent(output_path)
    reward = episode.get("reward", {})
    arrays = {
        "actions": np.asarray(episode["actions"], dtype=np.float32),
        "dones": np.asarray(episode["dones"], dtype=bool),
        "env_rewards": np.asarray(episode.get("env_rewards", []), dtype=np.float32),
        "eef_positions": np.asarray(episode["state_trace"]["eef_positions"], dtype=np.float32),
        "object_positions": np.asarray(episode["state_trace"]["object_positions"], dtype=np.float32),
        "target_positions": np.asarray(episode["state_trace"]["target_positions"], dtype=np.float32),
        "gripper_positions": np.asarray(episode["state_trace"]["gripper_positions"], dtype=np.float32),
        "agentview_images": np.asarray(episode["observations"]["agentview_images"], dtype=np.uint8),
        "robot0_eye_in_hand_images": np.asarray(
            episode["observations"]["robot0_eye_in_hand_images"], dtype=np.uint8
        ),
        "raw_observations_json": _to_np_object(episode["observations"]["raw_observations_json"]),
        "metadata_json": np.array(json.dumps(episode["metadata"], ensure_ascii=False), dtype=object),
        "reward_json": np.array(json.dumps(reward, ensure_ascii=False), dtype=object),
    }
    np.savez_compressed(output_path, **arrays)
    return Path(output_path)


def load_episode_npz(path: str | Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    metadata = json.loads(str(loaded["metadata_json"].item()))
    reward = json.loads(str(loaded["reward_json"].item()))
    raw_observations_json = [str(item) for item in loaded["raw_observations_json"].tolist()]
    return {
        "metadata": metadata,
        "reward": reward,
        "actions": loaded["actions"].astype(np.float32),
        "dones": loaded["dones"].astype(bool),
        "env_rewards": loaded["env_rewards"].astype(np.float32),
        "state_trace": {
            "eef_positions": loaded["eef_positions"].astype(np.float32),
            "object_positions": loaded["object_positions"].astype(np.float32),
            "target_positions": loaded["target_positions"].astype(np.float32),
            "gripper_positions": loaded["gripper_positions"].astype(np.float32),
        },
        "observations": {
            "agentview_images": loaded["agentview_images"].astype(np.uint8),
            "robot0_eye_in_hand_images": loaded["robot0_eye_in_hand_images"].astype(np.uint8),
            "raw_observations_json": raw_observations_json,
        },
    }


def compute_episode_payload_hash(episode: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(episode["actions"]).tobytes())
    digest.update(np.asarray(episode["dones"]).tobytes())
    digest.update(np.asarray(episode["state_trace"]["eef_positions"]).tobytes())
    digest.update(np.asarray(episode["state_trace"]["object_positions"]).tobytes())
    digest.update(np.asarray(episode["state_trace"]["target_positions"]).tobytes())
    digest.update(np.asarray(episode["state_trace"]["gripper_positions"]).tobytes())
    digest.update(np.asarray(episode["observations"]["agentview_images"]).tobytes())
    digest.update(np.asarray(episode["observations"]["robot0_eye_in_hand_images"]).tobytes())
    digest.update(json.dumps(episode["metadata"], sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return Path(path)


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = ensure_parent(path)
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return Path(path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows
