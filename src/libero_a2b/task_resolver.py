from __future__ import annotations

from dataclasses import dataclass
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


def _require_libero():
    try:
        from libero.libero import benchmark
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO is not installed. Install LIBERO and its simulation dependencies "
            "before running collection."
        ) from exc
    return benchmark


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
                return ResolvedTask(
                    family=cfg.task.family,
                    task_id=task_id,
                    task_name=getattr(task, "name", f"task_{task_id}"),
                    task_language=language,
                    movable_object_name=cfg.task.movable_object_name,
                    movable_object_aliases=cfg.task.movable_object_aliases,
                    reference_object_name=cfg.task.reference_object_name,
                    reference_object_aliases=cfg.task.reference_object_aliases,
                )
        raise ValueError(f"Configured task_id={cfg.task.task_id} not found in {cfg.task.family}.")

    if cfg.task.task_name is not None:
        lowered = cfg.task.task_name.lower()
        for task_id, task, language in tasks:
            if lowered in language.lower() or lowered == getattr(task, "name", "").lower():
                return ResolvedTask(
                    family=cfg.task.family,
                    task_id=task_id,
                    task_name=getattr(task, "name", f"task_{task_id}"),
                    task_language=language,
                    movable_object_name=cfg.task.movable_object_name,
                    movable_object_aliases=cfg.task.movable_object_aliases,
                    reference_object_name=cfg.task.reference_object_name,
                    reference_object_aliases=cfg.task.reference_object_aliases,
                )
        raise ValueError(f"Configured task_name={cfg.task.task_name!r} not found in {cfg.task.family}.")

    preferred = [item.lower() for item in cfg.task.preferred_language_matches]
    forbidden = [item.lower() for item in cfg.task.forbidden_language_matches]
    scored: list[tuple[int, int, str, object]] = []
    for task_id, task, language in tasks:
        lowered = language.lower()
        movable_score = _score_aliases(language, cfg.task.movable_object_aliases)
        reference_score = _score_aliases(language, cfg.task.reference_object_aliases)
        if movable_score <= 0 or reference_score <= 0:
            continue
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
    if not scored or scored[0][0] <= 0:
        raise RuntimeError(
            "Could not automatically resolve a LIBERO task containing the requested movable "
            f"object {cfg.task.movable_object_name!r} and reference object {cfg.task.reference_object_name!r}."
        )

    _, task_id, language, task = scored[0]
    return ResolvedTask(
        family=cfg.task.family,
        task_id=task_id,
        task_name=getattr(task, "name", f"task_{task_id}"),
        task_language=language,
        movable_object_name=cfg.task.movable_object_name,
        movable_object_aliases=cfg.task.movable_object_aliases,
        reference_object_name=cfg.task.reference_object_name,
        reference_object_aliases=cfg.task.reference_object_aliases,
    )
