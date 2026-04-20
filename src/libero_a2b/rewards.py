from __future__ import annotations

import numpy as np

from .config import PipelineConfig


def _rtg(step_rewards: np.ndarray) -> np.ndarray:
    output = np.zeros_like(step_rewards, dtype=np.float32)
    running = 0.0
    for idx in range(len(step_rewards) - 1, -1, -1):
        running += float(step_rewards[idx])
        output[idx] = running
    return output


def build_binary_rewards(episode: dict, broadcast: bool) -> dict:
    step_count = int(len(episode["actions"]))
    if broadcast:
        step_reward = np.ones(step_count, dtype=np.float32)
    else:
        step_reward = np.zeros(step_count, dtype=np.float32)
        if step_count:
            step_reward[-1] = 1.0
    reward_components = np.zeros((step_count, 1), dtype=np.float32)
    reward_components[:, 0] = step_reward
    return {
        "reward_mode": "binary",
        "step_reward": step_reward.tolist(),
        "reward_component_names": ["task_success"],
        "reward_components": reward_components.tolist(),
        "cumulative_return": np.cumsum(step_reward, dtype=np.float32).tolist(),
        "return_to_go": _rtg(step_reward).tolist(),
    }


def build_shaped_rewards(cfg: PipelineConfig, episode: dict) -> dict:
    trace = episode["state_trace"]
    eef = np.asarray(trace["eef_positions"], dtype=np.float32)
    obj = np.asarray(trace["object_positions"], dtype=np.float32)
    tgt = np.asarray(trace["target_positions"], dtype=np.float32)
    grip = np.asarray(trace["gripper_positions"], dtype=np.float32)

    if len(obj) == 0:
        raise ValueError("Episode has no state trace.")

    object_start = obj[0]
    step_count = len(obj)
    reward_names = [
        "reach_object",
        "grasp_object",
        "lift_object",
        "move_toward_target",
        "arrive_near_target",
        "place_object",
        "task_success",
    ]
    components = np.zeros((step_count, len(reward_names)), dtype=np.float32)

    eef_to_object = np.linalg.norm(eef - obj, axis=1)
    target_distance = np.linalg.norm(obj[:, :2] - tgt[:, :2], axis=1)
    lifted_height = obj[:, 2] - object_start[2]
    grip_closed = np.clip(grip.reshape(-1), a_min=0.0, a_max=None)

    reach_signal = np.clip(1.0 - (eef_to_object / 0.12), 0.0, 1.0)
    grasp_signal = np.where((eef_to_object < 0.045) & (grip_closed > 0.0), 1.0, 0.0)
    lift_signal = np.clip((lifted_height - 0.02) / max(cfg.randomization.lift_height, 1e-6), 0.0, 1.0)
    progress = np.maximum(0.0, target_distance[0] - target_distance)
    move_signal = np.clip(progress / max(target_distance[0], 1e-6), 0.0, 1.0)
    arrive_signal = np.where(target_distance <= cfg.randomization.success_distance_xy, 1.0, 0.0)
    place_signal = np.where(
        (arrive_signal > 0.0) & (lifted_height <= cfg.randomization.success_height_tol), 1.0, 0.0
    )
    success_signal = np.zeros(step_count, dtype=np.float32)
    success_signal[-1] = 1.0

    components[:, 0] = reach_signal * 0.15
    components[:, 1] = grasp_signal * 0.15
    components[:, 2] = lift_signal * 0.15
    components[:, 3] = move_signal * 0.2
    components[:, 4] = arrive_signal * 0.1
    components[:, 5] = place_signal * 0.1
    components[:, 6] = success_signal * 0.15
    step_reward = components.sum(axis=1, dtype=np.float32)

    return {
        "reward_mode": "dense",
        "step_reward": step_reward.astype(np.float32).tolist(),
        "reward_component_names": reward_names,
        "reward_components": components.astype(np.float32).tolist(),
        "cumulative_return": np.cumsum(step_reward, dtype=np.float32).tolist(),
        "return_to_go": _rtg(step_reward).tolist(),
    }
