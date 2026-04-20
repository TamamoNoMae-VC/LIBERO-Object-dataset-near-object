from __future__ import annotations

from collections import Counter

import numpy as np

from .config import PipelineConfig
from .storage import compute_episode_payload_hash, load_episode_npz, read_jsonl


def validate_raw_dataset(cfg: PipelineConfig) -> dict:
    index_path = cfg.resolve_path(cfg.paths.raw_dir) / "index.jsonl"
    rows = read_jsonl(index_path)
    if not rows:
        raise RuntimeError(f"No raw dataset index found at {index_path}")

    relation_pairs = Counter()
    trajectory_lengths = []
    payload_hashes = []
    for row in rows:
        if not row["success"]:
            raise AssertionError("Raw dataset contains a failed episode, which is forbidden.")
        if row["relation_a"] == row["relation_b"] and np.allclose(row["a_position"], row["b_position"]):
            raise AssertionError("A and B positions must differ for every saved episode.")
        relation_pairs[(row["relation_a"], row["relation_b"])] += 1
        trajectory_lengths.append(int(row["trajectory_length"]))
        episode = load_episode_npz(row["episode_path"])
        payload_hashes.append(compute_episode_payload_hash(episode))

    return {
        "episode_count": len(rows),
        "all_success": True,
        "avg_trajectory_length": float(np.mean(trajectory_lengths)),
        "min_trajectory_length": int(np.min(trajectory_lengths)),
        "max_trajectory_length": int(np.max(trajectory_lengths)),
        "unique_payload_hashes": len(set(payload_hashes)),
        "relation_pair_histogram": {
            f"{left}->{right}": count for (left, right), count in sorted(relation_pairs.items())
        },
    }


def validate_reward_exports(cfg: PipelineConfig) -> dict:
    raw_rows = read_jsonl(cfg.resolve_path(cfg.paths.raw_dir) / "index.jsonl")
    binary_rows = read_jsonl(cfg.resolve_path(cfg.paths.binary_dir) / "index.jsonl")
    shaped_rows = read_jsonl(cfg.resolve_path(cfg.paths.shaped_dir) / "index.jsonl")
    if not raw_rows or not binary_rows or not shaped_rows:
        raise RuntimeError("Missing raw or derived dataset indexes.")

    if len(raw_rows) != len(binary_rows) or len(raw_rows) != len(shaped_rows):
        raise AssertionError("Raw, binary, and shaped indexes must have identical lengths.")

    checks = []
    for raw_row, binary_row, shaped_row in zip(raw_rows, binary_rows, shaped_rows):
        for key in ("episode_id", "master_seed", "trajectory_length", "payload_hash"):
            if raw_row[key] != binary_row[key] or raw_row[key] != shaped_row[key]:
                raise AssertionError(f"Mismatch for {key} in episode {raw_row['episode_id']}.")
        checks.append(raw_row["episode_id"])

    return {
        "episode_count": len(checks),
        "matched_episode_ids": checks[:10],
        "all_payloads_match": True,
    }


def validate_single_master_seed(cfg: PipelineConfig) -> dict:
    raw_rows = read_jsonl(cfg.resolve_path(cfg.paths.raw_dir) / "index.jsonl")
    if not raw_rows:
        raise RuntimeError("No raw episodes found.")
    seeds = {row["master_seed"] for row in raw_rows}
    if seeds != {cfg.collection.master_seed}:
        raise AssertionError("Raw dataset contains multiple master seeds.")
    positions = {(tuple(row["a_position"]), tuple(row["b_position"])) for row in raw_rows}
    if len(positions) <= 1:
        raise AssertionError("Single-seed collection did not produce diverse A/B positions.")
    return {
        "master_seed": cfg.collection.master_seed,
        "unique_position_pairs": len(positions),
    }
