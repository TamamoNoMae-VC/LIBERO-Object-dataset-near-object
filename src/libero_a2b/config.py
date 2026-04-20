from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TaskConfig:
    family: str
    preferred_language_matches: list[str]
    forbidden_language_matches: list[str]
    custom_task_name: str
    movable_object_name: str
    movable_object_aliases: list[str]
    reference_object_name: str
    reference_object_aliases: list[str]
    task_name: str | None = None
    task_id: int | None = None


@dataclass(slots=True)
class CollectionConfig:
    master_seed: int
    successful_episode_count: int
    max_attempts: int
    max_steps_per_episode: int
    settle_steps: int
    grasp_close_steps: int
    release_open_steps: int
    post_grasp_lift_steps: int
    keep_failure_logs: bool
    save_rollouts: bool
    save_failed_rollouts: bool


@dataclass(slots=True)
class RandomizationConfig:
    allowed_relations: list[str]
    radius_min: float
    radius_max: float
    min_reference_distance: float
    basket_exclusion_radius: float
    camera_x_min: float
    camera_x_max: float
    camera_y_min: float
    camera_y_max: float
    upright_quat_dot_min: float
    hover_height: float
    grasp_height_offset: float
    lift_height: float
    placement_height_offset: float
    min_a_b_distance: float
    max_resample_attempts: int
    success_distance_xy: float
    success_height_tol: float
    max_action_delta: float
    close_gripper_value: float
    open_gripper_value: float
    orientation_action: list[float]
    grasp_orientation_action: list[float]
    transport_orientation_action: list[float]
    milk_grasp_local_offset: list[float]


@dataclass(slots=True)
class PathsConfig:
    root: str
    data_dir: str
    raw_dir: str
    binary_dir: str
    shaped_dir: str
    results_dir: str
    metrics_dir: str
    plots_dir: str
    reports_dir: str
    rollouts_dir: str


@dataclass(slots=True)
class ExportConfig:
    binary_variant_dir: str
    dense_variant_dir: str
    include_full_observations: bool
    binary_reward_broadcast: bool


@dataclass(slots=True)
class PipelineConfig:
    task: TaskConfig
    collection: CollectionConfig
    randomization: RandomizationConfig
    paths: PathsConfig
    export: ExportConfig
    config_path: Path
    config_name: str = field(init=False)

    def __post_init__(self) -> None:
        self.config_name = self.config_path.name

    @property
    def root_dir(self) -> Path:
        return self.config_path.parent.parent.resolve()

    def resolve_path(self, value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (self.root_dir / candidate).resolve()

    def ensure_output_dirs(self) -> None:
        for key in (
            "data_dir",
            "raw_dir",
            "binary_dir",
            "shaped_dir",
            "results_dir",
            "metrics_dir",
            "plots_dir",
            "reports_dir",
            "rollouts_dir",
        ):
            self.resolve_path(getattr(self.paths, key)).mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.__dict__,
            "collection": self.collection.__dict__,
            "randomization": self.randomization.__dict__,
            "paths": self.paths.__dict__,
            "export": self.export.__dict__,
            "config_path": str(self.config_path),
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    data = _load_yaml(config_path)
    task_cfg = dict(data["task"])
    task_cfg.setdefault("forbidden_language_matches", [])
    task_cfg.setdefault("custom_task_name", "orange_juice_near_basket_a2b")
    task_cfg.setdefault("movable_object_aliases", [task_cfg["movable_object_name"]])
    task_cfg.setdefault("reference_object_aliases", [task_cfg["reference_object_name"]])
    cfg = PipelineConfig(
        task=TaskConfig(**task_cfg),
        collection=CollectionConfig(**data["collection"]),
        randomization=RandomizationConfig(**data["randomization"]),
        paths=PathsConfig(**data["paths"]),
        export=ExportConfig(**data["export"]),
        config_path=config_path,
    )
    cfg.ensure_output_dirs()
    return cfg
