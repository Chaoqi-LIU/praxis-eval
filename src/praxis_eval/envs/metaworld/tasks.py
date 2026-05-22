# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""MetaWorld task definitions and eval-target inference."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, cast

MT50_GROUPS: tuple[str, ...] = ("easy", "medium", "hard", "very_hard")
MT50_ALIAS = "mt50"


@lru_cache(maxsize=1)
def _load_metaworld_config() -> dict[str, Any]:
    """Load LeRobot's MetaWorld task metadata without importing metaworld."""
    config_path = resources.files("lerobot.envs").joinpath("metaworld_config.json")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected MetaWorld config JSON to contain an object, got {type(data)!r}."
        )
    return cast(dict[str, Any], data)


def _task_descriptions() -> dict[str, str]:
    descriptions = _load_metaworld_config().get("TASK_DESCRIPTIONS")
    if not isinstance(descriptions, dict):
        raise TypeError("Expected TASK_DESCRIPTIONS to be a dict.")
    return {str(key): str(value) for key, value in descriptions.items()}


def _difficulty_to_tasks() -> dict[str, list[str]]:
    difficulty_to_tasks = _load_metaworld_config().get("DIFFICULTY_TO_TASKS")
    if not isinstance(difficulty_to_tasks, dict):
        raise TypeError("Expected DIFFICULTY_TO_TASKS to be a dict.")

    normalized: dict[str, list[str]] = {}
    for key, value in difficulty_to_tasks.items():
        if not isinstance(value, list):
            raise TypeError(f"Expected DIFFICULTY_TO_TASKS[{key!r}] to be a list.")
        normalized[str(key)] = [str(task) for task in value]
    return normalized


def canonicalize_task_selector(selector: str) -> str:
    """Normalize a MetaWorld task selector, group name, or leaf task name."""
    task = str(selector).strip()
    if task.startswith("metaworld-"):
        task = task[len("metaworld-") :]
    lower = task.lower()
    if lower == "mt-50":
        return MT50_ALIAS
    if lower == MT50_ALIAS:
        return MT50_ALIAS
    for group in MT50_GROUPS:
        if lower == group:
            return group
    return task


def expand_task_selectors(task: str) -> list[str]:
    """Expand comma-separated task selectors, replacing ``mt50`` with groups."""
    selectors = [
        canonicalize_task_selector(part)
        for part in str(task).split(",")
        if part.strip()
    ]
    if not selectors:
        raise ValueError("MetaWorld task selector must not be empty.")

    expanded: list[str] = []
    for selector in selectors:
        if selector == MT50_ALIAS:
            expanded.extend(MT50_GROUPS)
        else:
            expanded.append(selector)
    return expanded


def get_task_description(task_name: str) -> str:
    """Return the natural-language instruction for a MetaWorld leaf task."""
    task = canonicalize_task_selector(task_name)
    return _task_descriptions().get(task, task)


def resolve_task_name(task_group: str, task_id: int) -> str:
    """Resolve an eval-lane ``(task_group, task_id)`` pair to a leaf task."""
    group = canonicalize_task_selector(task_group)
    task_id = int(task_id)
    if group == MT50_ALIAS:
        raise ValueError(
            "Internal MetaWorld task groups should be expanded before eval."
        )

    difficulty_to_tasks = _difficulty_to_tasks()
    if group in difficulty_to_tasks:
        tasks = difficulty_to_tasks[group]
        if task_id < 0 or task_id >= len(tasks):
            raise ValueError(
                f"MetaWorld task_id {task_id} is out of range for group {group!r} "
                f"[0, {len(tasks) - 1}]."
            )
        return tasks[task_id]

    task = canonicalize_task_selector(group)
    descriptions = _task_descriptions()
    if task not in descriptions:
        available_groups = ", ".join([MT50_ALIAS, *MT50_GROUPS])
        raise ValueError(
            f"Unknown MetaWorld task or group {task_group!r}. "
            f"Known groups: {available_groups}."
        )
    if task_id != 0:
        raise ValueError(
            f"MetaWorld leaf task {task!r} supports only task_id=0, got {task_id}."
        )
    return task


def list_metaworld_tasks(
    raw_cfg: dict[str, Any],
    cfg_obj: Any,
    debug_verbose: bool = False,
) -> list[tuple[str, int]]:
    """List ``(task_group, task_id)`` pairs for the configured MetaWorld target."""
    _ = debug_verbose
    task_selector = str(getattr(cfg_obj, "task", raw_cfg.get("task", MT50_ALIAS)))
    difficulty_to_tasks = _difficulty_to_tasks()

    tasks: list[tuple[str, int]] = []
    for selector in expand_task_selectors(task_selector):
        if selector in difficulty_to_tasks:
            tasks.extend(
                (selector, task_id)
                for task_id in range(len(difficulty_to_tasks[selector]))
            )
        else:
            resolve_task_name(selector, 0)
            tasks.append((selector, 0))
    return tasks


def infer_metaworld_eval_target_from_dataset(
    dataset_name: str,
) -> tuple[str, str] | None:
    """Infer ``(env.type, env.task)`` from a MetaWorld dataset name."""
    if dataset_name == "metaworld":
        return "metaworld", MT50_ALIAS

    prefix = "metaworld_"
    if not dataset_name.startswith(prefix):
        return None
    selector = canonicalize_task_selector(dataset_name[len(prefix) :])
    if selector == MT50_ALIAS or selector in MT50_GROUPS:
        return "metaworld", selector

    try:
        resolve_task_name(selector, 0)
    except ValueError:
        return None
    return "metaworld", selector
