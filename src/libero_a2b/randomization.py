from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig


RELATION_TO_DIRECTION = {
    "left": np.array([-1.0, 0.0], dtype=np.float64),
    "right": np.array([1.0, 0.0], dtype=np.float64),
    "front": np.array([0.0, 1.0], dtype=np.float64),
    "back": np.array([0.0, -1.0], dtype=np.float64),
}


@dataclass(slots=True)
class PlacementSample:
    relation_a: str
    relation_b: str
    a_position: np.ndarray
    b_position: np.ndarray
    reference_position: np.ndarray


def make_episode_rng(master_seed: int, attempt_index: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{master_seed}:{attempt_index}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed)


def _sample_relation(rng: np.random.Generator, allowed_relations: list[str]) -> str:
    return str(rng.choice(allowed_relations))


def _sample_offset(rng: np.random.Generator, relation: str, radius_min: float, radius_max: float) -> np.ndarray:
    base = RELATION_TO_DIRECTION[relation]
    radius = float(rng.uniform(radius_min, radius_max))
    tangent = np.array([-base[1], base[0]], dtype=np.float64)
    tangent_noise = float(rng.uniform(-0.02, 0.02))
    return base * radius + tangent * tangent_noise


def _within_camera_bounds(cfg: PipelineConfig, position: np.ndarray) -> bool:
    return bool(
        cfg.randomization.camera_x_min <= float(position[0]) <= cfg.randomization.camera_x_max
        and cfg.randomization.camera_y_min <= float(position[1]) <= cfg.randomization.camera_y_max
    )


def _outside_reference_exclusion(cfg: PipelineConfig, position: np.ndarray, reference_position: np.ndarray) -> bool:
    reference_distance = float(np.linalg.norm(position[:2] - reference_position[:2]))
    return reference_distance >= max(cfg.randomization.min_reference_distance, cfg.randomization.basket_exclusion_radius)


def sample_a_b_positions(
    cfg: PipelineConfig,
    reference_position: np.ndarray,
    object_height: float,
    attempt_index: int,
) -> PlacementSample:
    rng = make_episode_rng(cfg.collection.master_seed, attempt_index)
    reference_position = np.asarray(reference_position, dtype=np.float64)
    relations = cfg.randomization.allowed_relations
    for _ in range(cfg.randomization.max_resample_attempts):
        relation_a = _sample_relation(rng, relations)
        relation_b = _sample_relation(rng, relations)
        if relation_a == relation_b:
            continue
        offset_a = _sample_offset(
            rng, relation_a, cfg.randomization.radius_min, cfg.randomization.radius_max
        )
        offset_b = _sample_offset(
            rng, relation_b, cfg.randomization.radius_min, cfg.randomization.radius_max
        )
        a_position = np.array(
            [reference_position[0] + offset_a[0], reference_position[1] + offset_a[1], object_height],
            dtype=np.float64,
        )
        b_position = np.array(
            [reference_position[0] + offset_b[0], reference_position[1] + offset_b[1], object_height],
            dtype=np.float64,
        )
        if not _outside_reference_exclusion(cfg, a_position, reference_position):
            continue
        if not _outside_reference_exclusion(cfg, b_position, reference_position):
            continue
        if not _within_camera_bounds(cfg, a_position):
            continue
        if not _within_camera_bounds(cfg, b_position):
            continue
        if np.linalg.norm(a_position[:2] - b_position[:2]) < cfg.randomization.min_a_b_distance:
            continue
        return PlacementSample(
            relation_a=relation_a,
            relation_b=relation_b,
            a_position=a_position,
            b_position=b_position,
            reference_position=reference_position.copy(),
        )
    raise RuntimeError(
        "Failed to sample valid A/B positions within the configured resample budget."
    )
