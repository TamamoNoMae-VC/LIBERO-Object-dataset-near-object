from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

from .config import PipelineConfig
from .randomization import sample_a_b_positions
from .storage import append_jsonl, compute_episode_payload_hash, save_episode_npz, write_json
from .task_resolver import ResolvedTask, resolve_task


REQUIRED_MODULES = ["libero", "robosuite", "h5py", "tensorflow", "torch", "yaml", "imageio"]
REQUIRED_ROBOSUITE_SYMBOL = "robosuite.environments.manipulation.single_arm_env"


def _safe_has_spec(module_name: str) -> tuple[bool, str | None]:
    try:
        return importlib.util.find_spec(module_name) is not None, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_preflight() -> dict[str, Any]:
    status: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "modules": {},
        "errors": {},
    }
    for name in REQUIRED_MODULES:
        present, error = _safe_has_spec(name)
        status["modules"][name] = present
        if error is not None:
            status["errors"][name] = error
    present, error = _safe_has_spec(REQUIRED_ROBOSUITE_SYMBOL)
    status["modules"][REQUIRED_ROBOSUITE_SYMBOL] = present
    if error is not None:
        status["errors"][REQUIRED_ROBOSUITE_SYMBOL] = error
    return status


def assert_preflight() -> None:
    status = run_preflight()
    modules = status["modules"]
    errors = status["errors"]
    missing = sorted(name for name, present in modules.items() if not present)
    if missing:
        if REQUIRED_ROBOSUITE_SYMBOL in missing and modules.get("robosuite", False):
            raise RuntimeError(
                "Installed robosuite is incompatible with LIBERO. "
                "LIBERO expects `robosuite.environments.manipulation.single_arm_env`, "
                "which is missing in newer robosuite releases. "
                "Use the LIBERO-compatible robosuite branch, for example:\n"
                "git clone https://github.com/ARISE-Initiative/robosuite.git\n"
                "cd robosuite\n"
                "git checkout v1.4.1_libero\n"
                "pip install -r requirements.txt\n"
                "pip install -r requirements-extra.txt\n"
                "pip install -e .\n"
                "python /content/datasetlibero2/scripts/patch_colab_compat.py --robosuite-root /content/datasetlibero2/robosuite\n"
                f"Underlying import error: {errors.get(REQUIRED_ROBOSUITE_SYMBOL, 'unknown')}"
            )
        raise RuntimeError(
            "Missing or incompatible dependencies for LIBERO collection: "
            + ", ".join(missing)
            + (
                "\nDetailed import errors: "
                + json.dumps(errors, ensure_ascii=False, indent=2)
                if errors
                else ""
            )
        )


@dataclass(slots=True)
class AttemptResult:
    success: bool
    attempt_index: int
    episode: dict[str, Any] | None
    failure_reason: str | None = None
    diagnostics: dict[str, Any] | None = None


def _get_libero_bits():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, get_libero_path, OffScreenRenderEnv


@contextmanager
def _libero_torch_load_compat():
    import torch

    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


def _get_initial_state(task_suite, task_id: int):
    with _libero_torch_load_compat():
        initial_states = task_suite.get_task_init_states(task_id)
    return initial_states[0]


def _build_env(resolved: ResolvedTask):
    benchmark, get_libero_path, OffScreenRenderEnv = _get_libero_bits()
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[resolved.family]()
    task = task_suite.get_task(resolved.task_id)
    initial_state = _get_initial_state(task_suite, resolved.task_id)
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(0)
    env.reset()
    env.set_init_state(initial_state)
    return env


def _candidate_names(names: list[str], needle: str) -> list[str]:
    lowered = needle.lower()
    normalized = re.sub(r"[^a-z0-9]+", "", lowered)
    candidates = []
    for name in names:
        name_lower = name.lower()
        name_normalized = re.sub(r"[^a-z0-9]+", "", name_lower)
        if lowered in name_lower or (normalized and normalized in name_normalized):
            candidates.append(name)
    return candidates


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "task"


def _resolve_body_name(env, aliases: list[str], entity_label: str) -> str:
    body_names = list(env.sim.model.body_names)
    candidates = []
    for alias in aliases:
        candidates.extend(_candidate_names(body_names, alias))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise RuntimeError(
            f"Could not resolve body for {entity_label}. "
            f"Aliases tried: {aliases!r}. "
            f"Available body names sample: {body_names[:120]!r}"
        )
    return candidates[0]


def _resolve_joint_name(env, aliases: list[str], entity_label: str) -> str:
    joint_names = list(env.sim.model.joint_names)
    candidates = []
    for alias in aliases:
        candidates.extend(_candidate_names(joint_names, alias))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise RuntimeError(
            f"Could not resolve free joint for {entity_label}. "
            f"Aliases tried: {aliases!r}. "
            f"Available joint names sample: {joint_names[:120]!r}"
        )
    return candidates[0]


def _get_body_pos(env, body_name: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(body_name)
    return np.array(env.sim.data.body_xpos[body_id], dtype=np.float64)


def _get_joint_qpos(env, joint_name: str) -> np.ndarray:
    return np.asarray(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        # MuJoCo free-joint quaternions are stored as [w, x, y, z].
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    # MuJoCo quaternion layout: [w, x, y, z].
    w1, x1, y1, z1 = _normalize_quat(lhs)
    w2, x2, y2, z2 = _normalize_quat(rhs)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _rotate_vector_by_quat(vector: np.ndarray, quat: np.ndarray) -> np.ndarray:
    # Rotate a 3D vector by a MuJoCo quaternion without normalizing the vector magnitude.
    q = _normalize_quat(quat)
    w = float(q[0])
    xyz = np.asarray(q[1:4], dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    uv = np.cross(xyz, vector)
    uuv = np.cross(xyz, uv)
    return vector + 2.0 * (w * uv + uuv)


def _quat_alignment_dot(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_n = _normalize_quat(lhs)
    rhs_n = _normalize_quat(rhs)
    return float(abs(np.dot(lhs_n, rhs_n)))


def _within_camera_bounds(cfg: PipelineConfig, position: np.ndarray) -> bool:
    return bool(
        cfg.randomization.camera_x_min <= float(position[0]) <= cfg.randomization.camera_x_max
        and cfg.randomization.camera_y_min <= float(position[1]) <= cfg.randomization.camera_y_max
    )


def _set_object_pose(env, joint_name: str, position: np.ndarray, quat: np.ndarray | None = None) -> None:
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if quat is None else quat
    qpos = np.concatenate([np.asarray(position, dtype=np.float64), np.asarray(quat, dtype=np.float64)])
    env.sim.data.set_joint_qpos(joint_name, qpos)
    try:
        env.sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    except Exception:
        pass
    env.sim.forward()


def _step_noop(env, steps: int) -> Any:
    obs = None
    for _ in range(steps):
        obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    return obs


def _extract_gripper(obs: dict[str, Any]) -> float:
    raw = np.asarray(obs.get("robot0_gripper_qpos", [0.0]), dtype=np.float64).reshape(-1)
    return float(raw.mean()) if raw.size else 0.0


def _move_delta(current: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    delta = target - current
    clipped = np.clip(delta, -max_delta, max_delta)
    return clipped.astype(np.float64)


def _get_controller_translation_scale(env) -> np.ndarray:
    fallback = np.full(3, 0.05, dtype=np.float64)
    try:
        robots = getattr(env, "robots", None)
        if not robots:
            return fallback
        controller = getattr(robots[0], "controller", None)
        if controller is None:
            return fallback
        output_max = np.asarray(getattr(controller, "output_max", []), dtype=np.float64).reshape(-1)
        input_max = np.asarray(getattr(controller, "input_max", []), dtype=np.float64).reshape(-1)
        if output_max.size < 3 or input_max.size < 3:
            return fallback
        scale = np.divide(
            np.abs(output_max[:3]),
            np.abs(input_max[:3]),
            out=fallback.copy(),
            where=np.abs(input_max[:3]) > 1e-6,
        )
        scale[~np.isfinite(scale)] = fallback[~np.isfinite(scale)]
        return np.clip(scale, 1e-4, 0.25)
    except Exception:
        return fallback


def _world_delta_to_action_delta(
    current: np.ndarray,
    target: np.ndarray,
    max_delta: float,
    translation_scale: np.ndarray,
) -> np.ndarray:
    world_delta = _move_delta(current, target, max_delta)
    normalized = np.divide(
        world_delta,
        translation_scale,
        out=np.zeros(3, dtype=np.float64),
        where=np.abs(translation_scale) > 1e-6,
    )
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _custom_success(
    cfg: PipelineConfig,
    object_position: np.ndarray,
    target_position: np.ndarray,
    reference_position: np.ndarray,
) -> bool:
    xy_close = np.linalg.norm(object_position[:2] - target_position[:2]) <= cfg.randomization.success_distance_xy
    z_close = abs(float(object_position[2] - target_position[2])) <= cfg.randomization.success_height_tol
    reference_distance = float(np.linalg.norm(object_position[:2] - reference_position[:2]))
    outside_basket = reference_distance >= cfg.randomization.basket_exclusion_radius
    visible = _within_camera_bounds(cfg, object_position)
    return bool(xy_close and z_close and outside_basket and visible)


def _move_until(
    env,
    obs: dict[str, Any],
    target_position: np.ndarray,
    gripper_value: float,
    max_delta: float,
    max_steps: int,
    orientation_action: list[float],
    position_tolerance: float = 0.015,
):
    collected = []
    translation_scale = _get_controller_translation_scale(env)
    orientation = np.asarray(orientation_action, dtype=np.float32)
    for _ in range(max_steps):
        eef_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        delta = _world_delta_to_action_delta(eef_position, target_position, max_delta, translation_scale)
        action = np.concatenate(
            [
                delta,
                orientation,
                np.array([gripper_value], dtype=np.float32),
            ]
        ).astype(np.float32)
        next_obs, reward, done, info = env.step(action.tolist())
        collected.append((action, next_obs, float(reward), bool(done), info))
        obs = next_obs
        if np.linalg.norm(target_position - np.asarray(obs["robot0_eef_pos"], dtype=np.float64)) <= position_tolerance:
            break
    return obs, collected


def _hold_gripper(env, obs: dict[str, Any], gripper_value: float, steps: int, orientation_action: list[float]):
    collected = []
    orientation = np.asarray(orientation_action, dtype=np.float32)
    for _ in range(steps):
        action = np.concatenate(
            [
                np.zeros(3, dtype=np.float32),
                orientation,
                np.array([gripper_value], dtype=np.float32),
            ]
        ).astype(np.float32)
        next_obs, reward, done, info = env.step(action.tolist())
        collected.append((action, next_obs, float(reward), bool(done), info))
        obs = next_obs
    return obs, collected


def _append_transition_buffers(
    env,
    transitions,
    observations: dict[str, list],
    state_trace: dict[str, list],
    actions: list,
    dones: list,
    env_rewards: list,
    movable_body: str,
    placement_target: np.ndarray,
):
    env_done_seen = False
    for action, next_obs, reward, done, _ in transitions:
        observations["agentview_images"].append(np.asarray(next_obs["agentview_image"], dtype=np.uint8))
        observations["robot0_eye_in_hand_images"].append(
            np.asarray(next_obs["robot0_eye_in_hand_image"], dtype=np.uint8)
        )
        observations["raw_observations_json"].append(_serialize_obs(next_obs))
        actions.append(action)
        env_rewards.append(float(reward))
        dones.append(bool(done))
        object_position = _get_body_pos(env, movable_body)
        state_trace["eef_positions"].append(np.asarray(next_obs["robot0_eef_pos"], dtype=np.float32))
        state_trace["object_positions"].append(object_position.astype(np.float32))
        state_trace["target_positions"].append(np.asarray(placement_target, dtype=np.float32))
        state_trace["gripper_positions"].append(np.array([_extract_gripper(next_obs)], dtype=np.float32))
        env_done_seen = env_done_seen or bool(done)
    return env_done_seen


def _compute_grasp_diagnostics(
    state_trace: dict[str, list],
    settled_object_position: np.ndarray,
) -> dict[str, Any]:
    eef_positions = np.asarray(state_trace["eef_positions"], dtype=np.float64)
    object_positions = np.asarray(state_trace["object_positions"], dtype=np.float64)
    gripper_positions = np.asarray(state_trace["gripper_positions"], dtype=np.float64).reshape(-1)
    if len(eef_positions) == 0 or len(object_positions) == 0:
        return {
            "min_eef_to_object_xyz": None,
            "min_eef_to_object_xy": None,
            "min_eef_height_over_object": None,
            "initial_eef_position": None,
            "final_eef_position": None,
            "max_gripper_position": None,
            "min_gripper_position": None,
            "final_gripper_position": None,
        }
    eef_to_object = eef_positions - object_positions
    eef_to_object_xyz = np.linalg.norm(eef_to_object, axis=1)
    eef_to_object_xy = np.linalg.norm(eef_to_object[:, :2], axis=1)
    settled_clearance = eef_positions[:, 2] - float(settled_object_position[2])
    return {
        "min_eef_to_object_xyz": float(eef_to_object_xyz.min()),
        "min_eef_to_object_xy": float(eef_to_object_xy.min()),
        "min_eef_height_over_object": float(settled_clearance.min()),
        "initial_eef_position": eef_positions[0].tolist(),
        "final_eef_position": eef_positions[-1].tolist(),
        "max_gripper_position": float(gripper_positions.max()) if gripper_positions.size else None,
        "min_gripper_position": float(gripper_positions.min()) if gripper_positions.size else None,
        "final_gripper_position": float(gripper_positions[-1]) if gripper_positions.size else None,
    }


def _serialize_obs(obs: dict[str, Any]) -> str:
    serializable = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            try:
                json.dumps(value)
                serializable[key] = value
            except TypeError:
                serializable[key] = str(value)
    return json.dumps(serializable, ensure_ascii=False)


def _write_env_name_debug(cfg: PipelineConfig, env, resolved: ResolvedTask, attempt_index: int) -> None:
    payload = {
        "attempt_index": attempt_index,
        "task_family": resolved.family,
        "task_id": resolved.task_id,
        "task_name": resolved.task_name,
        "task_language": resolved.task_language,
        "movable_object_aliases": resolved.movable_object_aliases,
        "reference_object_aliases": resolved.reference_object_aliases,
        "body_names": list(env.sim.model.body_names),
        "joint_names": list(env.sim.model.joint_names),
    }
    write_json(
        cfg.resolve_path(cfg.paths.reports_dir) / f"env_name_debug_attempt_{attempt_index:04d}.json",
        payload,
    )


def _execute_phase(
    cfg: PipelineConfig,
    env,
    obs: dict[str, Any],
    phase_name: str,
    target_position: np.ndarray,
    gripper_value: float,
    move_steps: int,
    hold_steps: int,
    orientation_action: list[float],
    observations: dict[str, list],
    state_trace: dict[str, list],
    actions: list,
    dones: list,
    env_rewards: list,
    movable_body: str,
    phase_log: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    env_done_seen = False
    obs, transitions = _move_until(
        env,
        obs,
        target_position,
        gripper_value=gripper_value,
        max_delta=cfg.randomization.max_action_delta,
        max_steps=min(move_steps, cfg.collection.max_steps_per_episode - len(actions)),
        orientation_action=orientation_action,
    )
    phase_log.append(
        {
            "phase": phase_name,
            "target_position": np.asarray(target_position, dtype=np.float64).tolist(),
            "transition_count": len(transitions),
            "hold_steps": hold_steps,
            "gripper_value": float(gripper_value),
        }
    )
    env_done_seen = _append_transition_buffers(
        env,
        transitions,
        observations,
        state_trace,
        actions,
        dones,
        env_rewards,
        movable_body,
        np.asarray(target_position, dtype=np.float64),
    ) or env_done_seen

    if hold_steps > 0 and len(actions) < cfg.collection.max_steps_per_episode:
        obs, transitions = _hold_gripper(
            env,
            obs,
            gripper_value,
            min(hold_steps, cfg.collection.max_steps_per_episode - len(actions)),
            orientation_action,
        )
        phase_log.append(
            {
                "phase": f"{phase_name}_hold",
                "target_position": np.asarray(target_position, dtype=np.float64).tolist(),
                "transition_count": len(transitions),
                "hold_steps": len(transitions),
                "gripper_value": float(gripper_value),
            }
        )
        env_done_seen = _append_transition_buffers(
            env,
            transitions,
            observations,
            state_trace,
            actions,
            dones,
            env_rewards,
            movable_body,
            np.asarray(target_position, dtype=np.float64),
        ) or env_done_seen
    return obs, env_done_seen


def _compute_milk_grasp_anchor(joint_qpos: np.ndarray, local_offset: np.ndarray) -> np.ndarray:
    object_position = np.asarray(joint_qpos[:3], dtype=np.float64)
    object_quat = np.asarray(joint_qpos[3:7], dtype=np.float64)
    return object_position + _rotate_vector_by_quat(local_offset, object_quat)


def _validate_spawn_state(
    cfg: PipelineConfig,
    placement,
    settled_object_position: np.ndarray,
    settled_object_quat: np.ndarray,
    object_spawn_quat: np.ndarray,
) -> tuple[bool, str | None, dict[str, Any]]:
    diagnostics = {
        "commanded_a_position": placement.a_position.tolist(),
        "settled_object_position": settled_object_position.tolist(),
        "reference_position": placement.reference_position.tolist(),
    }
    commanded_displacement_xy = float(np.linalg.norm(settled_object_position[:2] - placement.a_position[:2]))
    diagnostics["commanded_displacement_xy"] = commanded_displacement_xy
    if commanded_displacement_xy > 0.06:
        return False, "object_pose_drifted_after_spawn", diagnostics

    reference_distance = float(np.linalg.norm(settled_object_position[:2] - placement.reference_position[:2]))
    diagnostics["reference_distance_xy"] = reference_distance
    if reference_distance < cfg.randomization.basket_exclusion_radius:
        return False, "spawn_inside_basket_exclusion_zone", diagnostics

    if not _within_camera_bounds(cfg, settled_object_position):
        return False, "spawn_outside_camera_view", diagnostics

    upright_dot = _quat_alignment_dot(settled_object_quat, object_spawn_quat)
    diagnostics["upright_quat_dot"] = upright_dot
    if upright_dot < cfg.randomization.upright_quat_dot_min:
        return False, "spawn_not_upright", diagnostics

    return True, None, diagnostics


def _collect_single_attempt(
    cfg: PipelineConfig,
    resolved: ResolvedTask,
    attempt_index: int,
    saved_episode_id: int,
) -> AttemptResult:
    env = _build_env(resolved)
    try:
        try:
            movable_body = _resolve_body_name(env, resolved.movable_object_aliases, "movable object")
            reference_body = _resolve_body_name(env, resolved.reference_object_aliases, "reference object")
            movable_joint = _resolve_joint_name(env, resolved.movable_object_aliases, "movable object")
        except RuntimeError:
            _write_env_name_debug(cfg, env, resolved, attempt_index)
            raise

        if movable_body == reference_body:
            _write_env_name_debug(cfg, env, resolved, attempt_index)
            return AttemptResult(
                success=False,
                attempt_index=attempt_index,
                episode=None,
                failure_reason="movable_object_resolved_to_reference_object",
                diagnostics={
                    "resolved_movable_body": movable_body,
                    "resolved_reference_body": reference_body,
                    "resolved_movable_joint": movable_joint,
                },
            )

        _step_noop(env, cfg.collection.settle_steps)
        reference_position = _get_body_pos(env, reference_body)
        print(reference_position)
        movable_joint_qpos = _get_joint_qpos(env, movable_joint)
        object_spawn_position = movable_joint_qpos[:3].copy()
        object_spawn_quat = movable_joint_qpos[3:7].copy()
        placement = sample_a_b_positions(cfg, reference_position, float(object_spawn_position[2]), attempt_index)
        print(placement.a_position)
        _set_object_pose(env, movable_joint, placement.a_position, quat=object_spawn_quat)
        obs = _step_noop(env, cfg.collection.settle_steps)
        if obs is None:
            obs = env._get_observations()
        settled_object_position = _get_body_pos(env, movable_body)
        print(settled_object_position)
        settled_joint_qpos = _get_joint_qpos(env, movable_joint)
        controller_translation_scale = _get_controller_translation_scale(env)
        spawn_ok, spawn_failure_reason, spawn_diagnostics = _validate_spawn_state(
            cfg,
            placement,
            settled_object_position,
            settled_joint_qpos[3:7],
            object_spawn_quat,
        )
        if not spawn_ok:
            return AttemptResult(
                success=False,
                attempt_index=attempt_index,
                episode=None,
                failure_reason=spawn_failure_reason,
                diagnostics={
                    "resolved_movable_body": movable_body,
                    "resolved_reference_body": reference_body,
                    "resolved_movable_joint": movable_joint,
                    **spawn_diagnostics,
                },
            )

        observations = {
            "agentview_images": [],
            "robot0_eye_in_hand_images": [],
            "raw_observations_json": [],
        }
        state_trace = {
            "eef_positions": [],
            "object_positions": [],
            "target_positions": [],
            "gripper_positions": [],
        }
        actions = []
        dones = []
        env_rewards = []
        env_done_seen = False
        phase_log: list[dict[str, Any]] = []
        phase_specs = [
            {
                "name": "approach_above_object",
                "target_kind": "object_anchor",
                "z_offset": cfg.randomization.hover_height,
                "gripper_value": cfg.randomization.open_gripper_value,
                "move_steps": 35,
                "hold_steps": 0,
                "orientation_action": cfg.randomization.grasp_orientation_action,
            },
            {
                "name": "descend_to_object",
                "target_kind": "object_anchor",
                "z_offset": cfg.randomization.grasp_height_offset,
                "gripper_value": cfg.randomization.open_gripper_value,
                "move_steps": 30,
                "hold_steps": 0,
                "orientation_action": cfg.randomization.grasp_orientation_action,
            },
            {
                "name": "close_on_object",
                "target_kind": "object_anchor",
                "z_offset": cfg.randomization.grasp_height_offset,
                "gripper_value": cfg.randomization.close_gripper_value,
                "move_steps": 0,
                "hold_steps": cfg.collection.grasp_close_steps,
                "orientation_action": cfg.randomization.grasp_orientation_action,
            },
            {
                "name": "lift_object",
                "target_kind": "object_anchor",
                "z_offset": cfg.randomization.lift_height,
                "gripper_value": cfg.randomization.close_gripper_value,
                "move_steps": 35,
                "hold_steps": cfg.collection.post_grasp_lift_steps,
                "orientation_action": cfg.randomization.grasp_orientation_action,
            },
            {
                "name": "move_above_target",
                "target_kind": "target",
                "z_offset": cfg.randomization.lift_height,
                "gripper_value": cfg.randomization.close_gripper_value,
                "move_steps": 45,
                "hold_steps": 0,
                "orientation_action": cfg.randomization.transport_orientation_action,
            },
            {
                "name": "descend_to_target",
                "target_kind": "target",
                "z_offset": cfg.randomization.placement_height_offset,
                "gripper_value": cfg.randomization.close_gripper_value,
                "move_steps": 30,
                "hold_steps": 0,
                "orientation_action": cfg.randomization.transport_orientation_action,
            },
            {
                "name": "release_object",
                "target_kind": "target",
                "z_offset": cfg.randomization.placement_height_offset,
                "gripper_value": cfg.randomization.open_gripper_value,
                "move_steps": 0,
                "hold_steps": cfg.collection.release_open_steps,
                "orientation_action": cfg.randomization.transport_orientation_action,
            },
            {
                "name": "retreat_after_release",
                "target_kind": "target",
                "z_offset": cfg.randomization.hover_height,
                "gripper_value": cfg.randomization.open_gripper_value,
                "move_steps": 25,
                "hold_steps": 0,
                "orientation_action": cfg.randomization.transport_orientation_action,
            },
        ]

        for phase_spec in phase_specs:
            if len(actions) >= cfg.collection.max_steps_per_episode:
                break
            live_joint_qpos = _get_joint_qpos(env, movable_joint)
            live_object_position = np.asarray(live_joint_qpos[:3], dtype=np.float64)
            live_grasp_anchor = _compute_milk_grasp_anchor(
                live_joint_qpos,
                np.asarray(cfg.randomization.milk_grasp_local_offset, dtype=np.float64),
            )
            base_position = live_grasp_anchor if phase_spec["target_kind"] == "object_anchor" else placement.b_position
            target_position = np.array(
                [
                    base_position[0],
                    base_position[1],
                    base_position[2] + phase_spec["z_offset"],
                ],
                dtype=np.float64,
            )
            if phase_spec["move_steps"] <= 0 and phase_spec["hold_steps"] <= 0:
                continue
            obs, phase_done = _execute_phase(
                cfg,
                env,
                obs,
                phase_name=phase_spec["name"],
                target_position=target_position,
                gripper_value=phase_spec["gripper_value"],
                move_steps=phase_spec["move_steps"],
                hold_steps=phase_spec["hold_steps"],
                orientation_action=phase_spec["orientation_action"],
                observations=observations,
                state_trace=state_trace,
                actions=actions,
                dones=dones,
                env_rewards=env_rewards,
                movable_body=movable_body,
                phase_log=phase_log,
            )
            env_done_seen = env_done_seen or phase_done

        final_object_position = _get_body_pos(env, movable_body)
        object_positions_np = np.asarray(state_trace["object_positions"], dtype=np.float32)
        max_object_lift = (
            float(object_positions_np[:, 2].max() - settled_object_position[2]) if len(object_positions_np) else 0.0
        )
        final_target_distance = float(np.linalg.norm(final_object_position[:2] - placement.b_position[:2]))
        grasp_diagnostics = _compute_grasp_diagnostics(state_trace, settled_object_position)
        success = _custom_success(
            cfg,
            final_object_position,
            placement.b_position,
            placement.reference_position,
        )
        if not success:
            if max_object_lift < 0.03:
                failure_reason = "controller_failed_to_grasp_or_lift"
            elif final_target_distance > cfg.randomization.success_distance_xy:
                failure_reason = "scripted_controller_did_not_reach_target"
            else:
                failure_reason = "controller_failed_to_release_cleanly"
            failure_payload = {
                "attempt_index": attempt_index,
                "failure_reason": failure_reason,
                "task_family": resolved.family,
                "task_id": resolved.task_id,
                "custom_task_name": cfg.task.custom_task_name,
                "task_name": resolved.task_name,
                "task_language": resolved.task_language,
                "resolved_movable_body": movable_body,
                "resolved_reference_body": reference_body,
                "resolved_movable_joint": movable_joint,
                "relation_a": placement.relation_a,
                "relation_b": placement.relation_b,
                "settled_object_position": settled_object_position.tolist(),
                "object_spawn_position": object_spawn_position.tolist(),
                "object_spawn_quat": object_spawn_quat.tolist(),
                "settled_object_quat": settled_joint_qpos[3:7].tolist(),
                "milk_grasp_anchor": _compute_milk_grasp_anchor(
                    settled_joint_qpos,
                    np.asarray(cfg.randomization.milk_grasp_local_offset, dtype=np.float64),
                ).tolist(),
                "target_position": placement.b_position.tolist(),
                "controller_translation_scale": controller_translation_scale.tolist(),
                "max_object_lift": max_object_lift,
                "final_target_distance_xy": final_target_distance,
                "final_object_position": final_object_position.tolist(),
                "env_done_seen": env_done_seen,
                "phase_log": phase_log,
                "grasp_diagnostics": grasp_diagnostics,
            }
            failure_artifacts = _write_failed_attempt_artifacts(
                cfg,
                resolved,
                attempt_index,
                observations,
                failure_payload,
            )
            return AttemptResult(
                success=False,
                attempt_index=attempt_index,
                episode=None,
                failure_reason=failure_reason,
                diagnostics={
                    **failure_artifacts,
                    "controller_translation_scale": controller_translation_scale.tolist(),
                    "max_object_lift": max_object_lift,
                    "settled_object_position": settled_object_position.tolist(),
                    "final_target_distance_xy": final_target_distance,
                    "final_object_position": final_object_position.tolist(),
                    "target_position": placement.b_position.tolist(),
                    "env_done_seen": env_done_seen,
                    "phase_log": phase_log,
                    **grasp_diagnostics,
                },
            )

        episode = {
            "observations": observations,
            "actions": np.asarray(actions, dtype=np.float32),
            "env_rewards": np.asarray(env_rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "state_trace": state_trace,
            "metadata": {
                "episode_id": saved_episode_id,
                "attempt_index": attempt_index,
                "master_seed": cfg.collection.master_seed,
                "task_family": resolved.family,
                "task_id": resolved.task_id,
                "custom_task_name": cfg.task.custom_task_name,
                "task_name": resolved.task_name,
                "task_language": resolved.task_language,
                "resolved_movable_body": movable_body,
                "resolved_reference_body": reference_body,
                "resolved_movable_joint": movable_joint,
                "success": True,
                "relation_a": placement.relation_a,
                "relation_b": placement.relation_b,
                "a_position": placement.a_position.tolist(),
                "b_position": placement.b_position.tolist(),
                "object_spawn_position": object_spawn_position.tolist(),
                "object_spawn_quat": object_spawn_quat.tolist(),
                "object_pose_initial": settled_object_position.tolist(),
                "object_pose_commanded_initial": placement.a_position.tolist(),
                "object_pose_initial_quat": settled_joint_qpos[3:7].tolist(),
                "milk_grasp_anchor": _compute_milk_grasp_anchor(
                    settled_joint_qpos,
                    np.asarray(cfg.randomization.milk_grasp_local_offset, dtype=np.float64),
                ).tolist(),
                "object_pose_final": final_object_position.tolist(),
                "reference_object_pose": placement.reference_position.tolist(),
                "trajectory_length": len(actions),
                "env_done_seen": env_done_seen,
                "controller_translation_scale": controller_translation_scale.tolist(),
                "max_object_lift": max_object_lift,
                "phase_log": phase_log,
            },
        }
        return AttemptResult(
            success=True,
            attempt_index=attempt_index,
            episode=episode,
            diagnostics={
                "controller_translation_scale": controller_translation_scale.tolist(),
                "max_object_lift": max_object_lift,
                "settled_object_position": settled_object_position.tolist(),
                "final_target_distance_xy": final_target_distance,
                "final_object_position": final_object_position.tolist(),
                "target_position": placement.b_position.tolist(),
                "env_done_seen": env_done_seen,
                "phase_log": phase_log,
                **grasp_diagnostics,
            },
        )
    finally:
        env.close()


def _write_rollout_video(images: list[np.ndarray], output_path: Path) -> None:
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=20) as writer:
        for image in images:
            writer.append_data(image)


def _write_failed_attempt_artifacts(
    cfg: PipelineConfig,
    resolved: ResolvedTask,
    attempt_index: int,
    observations: dict[str, list],
    payload: dict[str, Any],
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    task_slug = _slugify(cfg.task.custom_task_name)
    report_path = cfg.resolve_path(cfg.paths.reports_dir) / f"failed_attempt_{attempt_index:04d}.json"
    write_json(report_path, payload)
    artifacts["failed_report_path"] = str(report_path.resolve())
    if cfg.collection.save_failed_rollouts and observations["agentview_images"]:
        rollout_path = (
            cfg.resolve_path(cfg.paths.rollouts_dir) / f"failed_attempt_{attempt_index:04d}_{task_slug}.mp4"
        )
        _write_rollout_video(observations["agentview_images"], rollout_path)
        artifacts["failed_rollout_path"] = str(rollout_path.resolve())
    return artifacts


def collect_successful_trajectories(cfg: PipelineConfig) -> dict[str, Any]:
    assert_preflight()
    resolved = resolve_task(cfg)
    task_slug = _slugify(cfg.task.custom_task_name)
    raw_root = cfg.resolve_path(cfg.paths.raw_dir)
    episodes_dir = raw_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    index_path = raw_root / "index.jsonl"
    if index_path.exists():
        index_path.unlink()

    failure_rows = []
    saved = 0
    collection_summary_path = cfg.resolve_path(cfg.paths.metrics_dir) / "collection_summary.json"
    resolved_task_summary = {
        "master_seed": cfg.collection.master_seed,
        "task_family": resolved.family,
        "custom_task_name": cfg.task.custom_task_name,
        "task_name": resolved.task_name,
        "task_language": resolved.task_language,
        "resolution_strategy": resolved.resolution_strategy,
        "task_score": resolved.task_score,
        "movable_object_name": resolved.movable_object_name,
        "movable_object_aliases": resolved.movable_object_aliases,
        "reference_object_name": resolved.reference_object_name,
        "reference_object_aliases": resolved.reference_object_aliases,
        "successful_episode_target": cfg.collection.successful_episode_count,
        "max_attempts": cfg.collection.max_attempts,
    }
    write_json(cfg.resolve_path(cfg.paths.reports_dir) / "resolved_task.json", resolved_task_summary)

    for attempt_index in range(cfg.collection.max_attempts):
        if saved >= cfg.collection.successful_episode_count:
            break
        print(
            f"[collect] attempt={attempt_index + 1}/{cfg.collection.max_attempts} "
            f"saved={saved}/{cfg.collection.successful_episode_count}",
            flush=True,
        )
        result = _collect_single_attempt(cfg, resolved, attempt_index, saved)
        if not result.success or result.episode is None:
            failure_row = {
                "attempt_index": attempt_index,
                "master_seed": cfg.collection.master_seed,
                "success": False,
                "failure_reason": result.failure_reason,
                "diagnostics": result.diagnostics or {},
            }
            failure_rows.append(failure_row)
            print(
                f"[collect] failed attempt={attempt_index + 1} reason={result.failure_reason} "
                f"diagnostics={json.dumps(failure_row['diagnostics'], ensure_ascii=False)}",
                flush=True,
            )
            continue

        episode = result.episode
        episode_path = episodes_dir / f"episode_{saved:06d}.npz"
        save_episode_npz(episode, episode_path)
        payload_hash = compute_episode_payload_hash(episode)
        row = {
            "episode_id": saved,
            "attempt_index": attempt_index,
            "master_seed": cfg.collection.master_seed,
            "task_family": resolved.family,
            "task_id": resolved.task_id,
            "custom_task_name": cfg.task.custom_task_name,
            "task_name": resolved.task_name,
            "task_language": resolved.task_language,
            "success": True,
            "relation_a": episode["metadata"]["relation_a"],
            "relation_b": episode["metadata"]["relation_b"],
            "a_position": episode["metadata"]["a_position"],
            "b_position": episode["metadata"]["b_position"],
            "trajectory_length": episode["metadata"]["trajectory_length"],
            "episode_path": str(episode_path.resolve()),
            "payload_hash": payload_hash,
        }
        append_jsonl(index_path, [row])
        if cfg.collection.save_rollouts:
            rollout_path = cfg.resolve_path(cfg.paths.rollouts_dir) / f"episode_{saved:06d}_{task_slug}.mp4"
            _write_rollout_video(episode["observations"]["agentview_images"], rollout_path)
        saved += 1
        print(
            f"[collect] saved episode_id={row['episode_id']} path={row['episode_path']}",
            flush=True,
        )

    if cfg.collection.keep_failure_logs:
        write_json(
            cfg.resolve_path(cfg.paths.metrics_dir) / "failed_attempts.json",
            {"failures": failure_rows},
        )

    summary = {
        "master_seed": cfg.collection.master_seed,
        "task_family": resolved.family,
        "custom_task_name": cfg.task.custom_task_name,
        "task_name": resolved.task_name,
        "task_language": resolved.task_language,
        "resolution_strategy": resolved.resolution_strategy,
        "task_score": resolved.task_score,
        "successful_episode_count": saved,
        "attempt_count": saved + len(failure_rows),
        "success_rate": float(saved / max(saved + len(failure_rows), 1)),
        "raw_index_path": str(index_path.resolve()),
    }
    write_json(collection_summary_path, summary)

    if saved != cfg.collection.successful_episode_count:
        raise RuntimeError(
            f"Collected only {saved} successful episodes, expected {cfg.collection.successful_episode_count}. "
            f"See {collection_summary_path} and "
            f"{cfg.resolve_path(cfg.paths.metrics_dir) / 'failed_attempts.json'} for details."
        )

    return summary
