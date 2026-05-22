# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa365 dataset selectors kept local to the RoboCasa family."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from praxis_eval.envs.robocasa.tasks import (
    get_subtasks,
    is_multitask,
    list_leaf_tasks,
)

ROBOCASA_DATASET_POOLS = (
    "pretrain-human",
    "pretrain-mimicgen",
    "target-human",
)

_POOL_TO_SPLIT_SOURCE: dict[str, tuple[str, str]] = {
    "pretrain-human": ("pretrain", "human"),
    "pretrain-mimicgen": ("pretrain", "mg"),
    "target-human": ("target", "human"),
}


@lru_cache(maxsize=1)
def _task_sets() -> dict[str, tuple[str, ...]]:
    from robocasa.utils import dataset_registry as dataset_registry_mod

    registry = getattr(dataset_registry_mod, "TASK_SET_REGISTRY", {})
    return {
        str(name): tuple(str(task_name) for task_name in task_names)
        for name, task_names in registry.items()
    }


@lru_cache(maxsize=1)
def _dataset_soups() -> dict[str, tuple[dict[str, Any], ...]]:
    from robocasa.utils import dataset_registry as dataset_registry_mod

    registry = getattr(dataset_registry_mod, "DATASET_SOUP_REGISTRY", {})
    return {
        str(name): tuple(dict(entry) for entry in entries)
        for name, entries in registry.items()
    }


def list_task_sets() -> list[str]:
    """Return RoboCasa365 task-set names such as ``atomic_seen``."""
    return list(_task_sets().keys())


def list_dataset_soups() -> list[str]:
    """Return RoboCasa365 dataset-soup names such as ``target_atomic_seen``."""
    return list(_dataset_soups().keys())


def list_dataset_pools() -> list[str]:
    """Return supported custom merge pools."""
    return list(ROBOCASA_DATASET_POOLS)


def pool_to_split_source(pool: str) -> tuple[str, str]:
    """Map a custom merge pool key to RoboCasa365 split/source selectors."""
    try:
        return _POOL_TO_SPLIT_SOURCE[str(pool)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown RoboCasa dataset pool {pool!r}. "
            f"Known: {', '.join(ROBOCASA_DATASET_POOLS)}."
        ) from exc


def resolve_dataset_soup_entries(dataset_soup: str) -> list[dict[str, Any]]:
    """Resolve an official RoboCasa365 dataset soup."""
    if dataset_soup not in _dataset_soups():
        raise ValueError(f"Unknown RoboCasa dataset soup: {dataset_soup}")
    return [dict(entry) for entry in _dataset_soups()[dataset_soup]]


def expand_custom_task_selector(
    *,
    tasks: list[str] | None = None,
    task_set: str | None = None,
    group: str | None = None,
) -> list[str]:
    """Resolve custom task selectors to RoboCasa365 leaf tasks."""
    selectors = sum(x is not None for x in (tasks, task_set, group))
    if selectors != 1:
        raise ValueError("Exactly one of tasks, task_set, or group must be provided.")

    if group is not None:
        if not is_multitask(group):
            raise ValueError(f"Unknown evaluator RoboCasa task group: {group}")
        return list(get_subtasks(group))

    if task_set is not None:
        if task_set not in _task_sets():
            raise ValueError(f"Unknown RoboCasa task set: {task_set}")
        return list(_task_sets()[task_set])

    assert tasks is not None
    known_leaf_tasks = set(list_leaf_tasks())
    expanded: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        for subtask in get_subtasks(task):
            if subtask not in known_leaf_tasks:
                raise ValueError(f"Unknown RoboCasa leaf task: {subtask}")
            if subtask not in seen:
                seen.add(subtask)
                expanded.append(subtask)
    return expanded


def resolve_pool_entries(
    *,
    tasks: list[str] | None = None,
    task_set: str | None = None,
    group: str | None = None,
    pool: str,
    demo_fraction: float = 1.0,
) -> list[dict[str, Any]]:
    """Resolve official RoboCasa365 dataset entries for one supported pool."""
    from robocasa.utils.dataset_registry_utils import get_ds_meta

    split, source = pool_to_split_source(pool)
    task_names = expand_custom_task_selector(
        tasks=tasks, task_set=task_set, group=group
    )

    entries: list[dict[str, Any]] = []
    for task_name in task_names:
        meta = get_ds_meta(
            task=task_name,
            split=split,
            source=source,
            demo_fraction=demo_fraction,
        )
        if meta is not None:
            entries.append(dict(meta))
    return entries
