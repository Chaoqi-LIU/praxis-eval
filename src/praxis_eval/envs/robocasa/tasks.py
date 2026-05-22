# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa365 task definitions and eval-target inference."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# Keep evaluator-specific multitask groups local here. Everything else should come
# from the installed RoboCasa task registry rather than duplicated leaf lists.
MT_TASKS: dict[str, list[str]] = {
    "mt5": [
        "CloseToasterOvenDoor",
        "OpenDrawer",
        "PickPlaceDrawerToCounter",
        "TurnOnElectricKettle",
        "SlideDishwasherRack",
    ]
}


@lru_cache(maxsize=1)
def _registered_env_names() -> tuple[str, ...]:
    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

    return tuple(str(name) for name in ALL_KITCHEN_ENVIRONMENTS)


@lru_cache(maxsize=1)
def _task_sets() -> dict[str, tuple[str, ...]]:
    from robocasa.utils import dataset_registry as dataset_registry_mod

    registry = getattr(dataset_registry_mod, "TASK_SET_REGISTRY", {})
    return {
        str(name): tuple(str(task_name) for task_name in task_names)
        for name, task_names in registry.items()
    }


@lru_cache(maxsize=1)
def _task_horizons() -> dict[str, int]:
    from robocasa.utils import dataset_registry as dataset_registry_mod

    horizons: dict[str, int] = {}
    for registry_name in ("ATOMIC_TASK_DATASETS", "COMPOSITE_TASK_DATASETS"):
        registry = getattr(dataset_registry_mod, registry_name, {})
        for task_name, meta in registry.items():
            horizon = meta.get("horizon")
            if horizon is not None:
                horizons[str(task_name)] = int(horizon)
    return horizons


@lru_cache(maxsize=1)
def list_leaf_tasks() -> list[str]:
    """Return RoboCasa365 leaf tasks known to the installed registry.

    We prefer dataset-registry order so downstream tools operate on the same task
    names the installed RoboCasa package advertises data for.
    """

    horizons = _task_horizons()
    registered = set(_registered_env_names())
    ordered = [task for task in horizons if task in registered]
    for task in _registered_env_names():
        if task not in horizons:
            ordered.append(task)
    return ordered


def is_multitask(task_name: str) -> bool:
    """Return True if task_name refers to a evaluator RoboCasa task group."""
    return task_name in MT_TASKS


def list_task_groups() -> list[str]:
    """Return evaluator-defined RoboCasa task-group keys."""
    return list(MT_TASKS)


def get_subtasks(task_name: str) -> list[str]:
    """Expand a task name (single or group) to a list of RoboCasa365 leaf tasks."""
    if is_multitask(task_name):
        return list(MT_TASKS[task_name])
    if task_name in _task_sets():
        return list(_task_sets()[task_name])
    return [task_name]


def get_task_horizon(task_name: str, *, default: int = 500) -> int:
    """Return the declared RoboCasa365 horizon for a task or task group."""
    horizons = _task_horizons()
    subtasks = get_subtasks(task_name)
    if not subtasks:
        return default
    return max(horizons.get(subtask, default) for subtask in subtasks)


def list_robocasa_tasks(
    raw_cfg: dict[str, Any],
    cfg_obj: Any,
    debug_verbose: bool = False,
) -> list[tuple[str, int]]:
    """List ``(task_name, task_id)`` pairs for the configured RoboCasa target."""

    _ = debug_verbose
    task = str(getattr(cfg_obj, "task", raw_cfg.get("task", "mt5")))
    subtasks = get_subtasks(task)
    return [(subtask, idx) for idx, subtask in enumerate(subtasks)]


def infer_robocasa_eval_target_from_dataset(
    dataset_name: str,
) -> tuple[str, str] | None:
    """Infer ``(env.type, env.task)`` from a RoboCasa dataset name."""

    prefix = "robocasa_"
    if not dataset_name.startswith(prefix):
        return None
    task = dataset_name[len(prefix) :]
    if task in MT_TASKS or task in _task_sets() or task in list_leaf_tasks():
        return "robocasa", task
    return None
