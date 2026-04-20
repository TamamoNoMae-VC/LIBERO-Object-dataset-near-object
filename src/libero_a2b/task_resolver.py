from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re

from .config import PipelineConfig


@dataclass(slots=True)
class ResolvedTask:
    family: str
    task_id: int
    task_name: str
    task_language: str
    movable_object_name: str
    movable_object_aliases: list[str]
    reference_object_name: str
    reference_object_aliases: list[str]
    resolution_strategy: str = "language"
    task_score: int | None = None


def _require_libero():
    try:
        from libero.libero import benchmark
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO is not installed. Install LIBERO and its simulation dependencies "
            "before running collection."
        ) from exc
    return benchmark


def _get_libero_env_bits():
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO environment modules are not available. Install LIBERO and its simulation "
            "dependencies before running collection."
        ) from exc
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


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _score_aliases(text: str, aliases: list[str]) -> int:
    lowered = text.lower()
    normalized_text = _normalize(text)
    score = 0
    for alias in aliases:
        alias_lower = alias.lower()
        alias_normalized = _normalize(alias)
        if alias_lower and alias_lower in lowered:
            score += 3
        if alias_normalized and alias_normalized in normalized_text:
            score += 2
    return score


def _normalized_contains(text: str, pattern: str) -> bool:
    normalized_text = _normalize(text)
    normalized_pattern = _normalize(pattern)
    return bool(normalized_pattern and normalized_pattern in normalized_text)


def _first_alias_index(text: str, aliases: list[str]) -> int | None:
    normalized_text = _normalize(text)
    best_index: int | None = None
    for alias in aliases:
        normalized_alias = _normalize(alias)
        if not normalized_alias:
            continue
        index = normalized_text.find(normalized_alias)
        if index >= 0 and (best_index is None or index < best_index):
            best_index = index
    return best_index


def _candidate_names(names: list[str], needle: str) -> list[str]:
    lowered = needle.lower()
    normalized = _normalize(needle)
    candidates = []
    for name in names:
        name_lower = name.lower()
        name_normalized = _normalize(name)
        if lowered in name_lower or (normalized and normalized in name_normalized):
            candidates.append(name)
    return candidates


def _find_alias_matches(names: list[str], aliases: list[str]) -> list[str]:
    matches: list[str] = []
    for alias in aliases:
        matches.extend(_candidate_names(names, alias))
    return list(dict.fromkeys(matches))


def _inspect_task_entities(family: str, task_id: int) -> dict[str, object]:
    benchmark, get_libero_path, OffScreenRenderEnv = _get_libero_env_bits()
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[family]()
    task = task_suite.get_task(task_id)
    with _libero_torch_load_compat():
        initial_states = task_suite.get_task_init_states(task_id)
    initial_state = initial_states[0]
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=256,
        camera_widths=256,
    )
    try:
        env.seed(0)
        env.reset()
        env.set_init_state(initial_state)
        return {
            "body_names": list(env.sim.model.body_names),
            "joint_names": list(env.sim.model.joint_names),
        }
    finally:
        env.close()


def _make_resolved_task(
    cfg: PipelineConfig,
    task_id: int,
    task,
    language: str,
    resolution_strategy: str,
    task_score: int | None,
) -> ResolvedTask:
    return ResolvedTask(
        family=cfg.task.family,
        task_id=task_id,
        task_name=getattr(task, "name", f"task_{task_id}"),
        task_language=language,
        movable_object_name=cfg.task.movable_object_name,
        movable_object_aliases=cfg.task.movable_object_aliases,
        reference_object_name=cfg.task.reference_object_name,
        reference_object_aliases=cfg.task.reference_object_aliases,
        resolution_strategy=resolution_strategy,
        task_score=task_score,
    )


def _validate_explicit_task_matches(cfg: PipelineConfig, resolved: ResolvedTask) -> ResolvedTask:
    inspection = _inspect_task_entities(resolved.family, resolved.task_id)
    body_names = list(inspection["body_names"])
    joint_names = list(inspection["joint_names"])
    movable_body_matches = _find_alias_matches(body_names, resolved.movable_object_aliases)
    movable_joint_matches = _find_alias_matches(joint_names, resolved.movable_object_aliases)
    reference_body_matches = _find_alias_matches(body_names, resolved.reference_object_aliases)
    if movable_body_matches and movable_joint_matches and reference_body_matches:
        return resolved
    raise RuntimeError(
        "Configured LIBERO task does not instantiate the requested movable/reference objects. "
        f"task_id={resolved.task_id}, task_name={resolved.task_name!r}, "
        f"task_language={resolved.task_language!r}, "
        f"movable aliases={resolved.movable_object_aliases!r}, "
        f"reference aliases={resolved.reference_object_aliases!r}, "
        f"movable body matches={movable_body_matches!r}, "
        f"movable joint matches={movable_joint_matches!r}, "
        f"reference body matches={reference_body_matches!r}, "
        f"body names sample={body_names[:40]!r}, "
        f"joint names sample={joint_names[:40]!r}"
    )


def resolve_task(cfg: PipelineConfig) -> ResolvedTask:
    benchmark = _require_libero()
    benchmark_dict = benchmark.get_benchmark_dict()
    if cfg.task.family not in benchmark_dict:
        raise ValueError(f"Unknown task family: {cfg.task.family}")

    task_suite = benchmark_dict[cfg.task.family]()
    tasks = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        language = getattr(task, "language", "").strip()
        tasks.append((task_id, task, language))

    if cfg.task.task_id is not None:
        for task_id, task, language in tasks:
            if task_id == cfg.task.task_id:
                return _validate_explicit_task_matches(
                    cfg,
                    _make_resolved_task(
                        cfg,
                        task_id,
                        task,
                        language,
                        resolution_strategy="explicit_task_id",
                        task_score=None,
                    ),
                )
        raise ValueError(f"Configured task_id={cfg.task.task_id} not found in {cfg.task.family}.")

    if cfg.task.task_name is not None:
        lowered = cfg.task.task_name.lower()
        for task_id, task, language in tasks:
            if lowered in language.lower() or lowered == getattr(task, "name", "").lower():
                return _validate_explicit_task_matches(
                    cfg,
                    _make_resolved_task(
                        cfg,
                        task_id,
                        task,
                        language,
                        resolution_strategy="explicit_task_name",
                        task_score=None,
                    ),
                )
        raise ValueError(f"Configured task_name={cfg.task.task_name!r} not found in {cfg.task.family}.")

    preferred = [item.lower() for item in cfg.task.preferred_language_matches]
    forbidden = [item.lower() for item in cfg.task.forbidden_language_matches]
    scored: list[tuple[int, int, str, object]] = []
    for task_id, task, language in tasks:
        lowered = language.lower()
        movable_score = _score_aliases(language, cfg.task.movable_object_aliases)
        reference_score = _score_aliases(language, cfg.task.reference_object_aliases)
        score = 0
        score += movable_score
        score += reference_score
        if "place" in lowered:
            score += 1
        if "near" in lowered or "side" in lowered or "next to" in lowered:
            score += 1
        movable_index = _first_alias_index(language, cfg.task.movable_object_aliases)
        reference_index = _first_alias_index(language, cfg.task.reference_object_aliases)
        if movable_index is not None and reference_index is not None:
            if movable_index < reference_index:
                score += 25
            else:
                score -= 25
        for idx, candidate in enumerate(preferred):
            if candidate in lowered or _normalized_contains(language, candidate):
                score += 100 - idx
        for candidate in forbidden:
            if candidate in lowered or _normalized_contains(language, candidate):
                score -= 1000
        scored.append((score, task_id, language, task))

    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    compatible_candidates: list[tuple[int, int, str, object, list[str], list[str], list[str]]] = []
    inspection_failures: list[str] = []
    for score, task_id, language, task in scored:
        try:
            inspection = _inspect_task_entities(cfg.task.family, task_id)
        except Exception as exc:
            inspection_failures.append(
                f"task_id={task_id} name={getattr(task, 'name', f'task_{task_id}')!r} "
                f"language={language!r} inspection_error={type(exc).__name__}: {exc}"
            )
            continue
        body_names = list(inspection["body_names"])
        joint_names = list(inspection["joint_names"])
        movable_body_matches = _find_alias_matches(body_names, cfg.task.movable_object_aliases)
        movable_joint_matches = _find_alias_matches(joint_names, cfg.task.movable_object_aliases)
        reference_body_matches = _find_alias_matches(body_names, cfg.task.reference_object_aliases)
        if movable_body_matches and movable_joint_matches and reference_body_matches:
            env_score = score + 20 + len(movable_body_matches) + len(movable_joint_matches) + len(reference_body_matches)
            compatible_candidates.append(
                (
                    env_score,
                    task_id,
                    language,
                    task,
                    movable_body_matches,
                    movable_joint_matches,
                    reference_body_matches,
                )
            )
            continue
        inspection_failures.append(
            f"task_id={task_id} name={getattr(task, 'name', f'task_{task_id}')!r} "
            f"language={language!r} movable_body_matches={movable_body_matches!r} "
            f"movable_joint_matches={movable_joint_matches!r} reference_body_matches={reference_body_matches!r} "
            f"body_names_sample={body_names[:20]!r} joint_names_sample={joint_names[:20]!r}"
        )

    compatible_candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    if not compatible_candidates:
        raise RuntimeError(
            "Could not automatically resolve a LIBERO task that both matches the requested "
            f"scene family {cfg.task.family!r} and instantiates the requested movable object "
            f"{cfg.task.movable_object_name!r} with reference object {cfg.task.reference_object_name!r}. "
            "Per-task inspection summary: "
            + " | ".join(inspection_failures[:10])
        )

    best_score, task_id, language, task, _, _, _ = compatible_candidates[0]
    return _make_resolved_task(
        cfg,
        task_id,
        task,
        language,
        resolution_strategy="env_validated_auto",
        task_score=best_score,
    )
