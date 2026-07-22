# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Official RoboCasa GR-1 tabletop task definitions."""

from __future__ import annotations

from typing import Any

_SUFFIX = "_GR1ArmsAndWaistFourierHands_Env"
_NAMESPACE = "gr1_unified"

ARTICULATED_TASKS: tuple[str, ...] = (
    "PnPCupToDrawerClose",
    "PnPPotatoToMicrowaveClose",
    "PnPMilkToMicrowaveClose",
    "PnPBottleToCabinetClose",
    "PnPWineToCabinetClose",
    "PnPCanToDrawerClose",
)
REARRANGEMENT_TASKS: tuple[str, ...] = (
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA",
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA",
    "PosttrainPnPNovelFromCuttingboardToPotSplitA",
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA",
    "PosttrainPnPNovelFromPlacematToBasketSplitA",
    "PosttrainPnPNovelFromPlacematToBowlSplitA",
    "PosttrainPnPNovelFromPlacematToPlateSplitA",
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA",
    "PosttrainPnPNovelFromPlateToBowlSplitA",
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA",
    "PosttrainPnPNovelFromPlateToPanSplitA",
    "PosttrainPnPNovelFromPlateToPlateSplitA",
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    "PosttrainPnPNovelFromTrayToPlateSplitA",
    "PosttrainPnPNovelFromTrayToPotSplitA",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA",
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA",
)


def _full_name(short_name: str) -> str:
    return f"{_NAMESPACE}/{short_name}{_SUFFIX}"


GR1_TASKS: tuple[str, ...] = tuple(
    _full_name(name) for name in (*ARTICULATED_TASKS, *REARRANGEMENT_TASKS)
)
TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "all": GR1_TASKS,
    "gr1_24": GR1_TASKS,
    "articulated_6": tuple(_full_name(name) for name in ARTICULATED_TASKS),
    "rearrangement_18": tuple(_full_name(name) for name in REARRANGEMENT_TASKS),
}
_SHORT_TO_FULL = {
    short_name: _full_name(short_name)
    for short_name in (*ARTICULATED_TASKS, *REARRANGEMENT_TASKS)
}
_CLASS_TO_FULL = {task.split("/", 1)[1]: task for task in GR1_TASKS}


def resolve_gr1_task(task: str) -> str:
    """Resolve a full Gym id, environment class name, or short task alias."""
    name = str(task).strip()
    if name in GR1_TASKS:
        return name
    if name in _SHORT_TO_FULL:
        return _SHORT_TO_FULL[name]
    if name in _CLASS_TO_FULL:
        return _CLASS_TO_FULL[name]
    raise ValueError(
        f"Unknown RoboCasa GR-1 task {task!r}. Use a full gr1_unified id, "
        "one of the 24 short task names, or a task group."
    )


def expand_gr1_tasks(task: str) -> list[str]:
    """Expand a task group or return one resolved leaf task."""
    name = str(task).strip()
    if name in TASK_GROUPS:
        return list(TASK_GROUPS[name])
    return [resolve_gr1_task(name)]


def list_robocasa_gr1_tasks(
    raw_cfg: dict[str, Any],
    cfg_obj: Any,
    debug_verbose: bool = False,
) -> list[tuple[str, int]]:
    """List evaluator ``(task_name, task_id)`` pairs."""
    _ = debug_verbose
    task = str(getattr(cfg_obj, "task", raw_cfg.get("task", "all")))
    task_names = expand_gr1_tasks(task)
    indexed_tasks = list(enumerate(task_names))
    selected_ids = getattr(cfg_obj, "task_ids", raw_cfg.get("task_ids"))
    if selected_ids is not None:
        ids = [int(task_id) for task_id in selected_ids]
        invalid = [task_id for task_id in ids if not 0 <= task_id < len(task_names)]
        if invalid:
            raise ValueError(
                f"RoboCasa GR-1 task_ids out of range for {task!r}: {invalid}."
            )
        indexed_tasks = [indexed_tasks[task_id] for task_id in ids]
    return [(task_name, task_id) for task_id, task_name in indexed_tasks]


def infer_robocasa_gr1_eval_target_from_dataset(
    dataset_name: str,
) -> tuple[str, str] | None:
    """Infer a GR-1 eval target from ``robocasa_gr1_<task>``."""
    prefix = "robocasa_gr1_"
    if not dataset_name.startswith(prefix):
        return None
    task = dataset_name[len(prefix) :]
    if task in TASK_GROUPS:
        return "robocasa_gr1", task
    try:
        resolve_gr1_task(task)
    except ValueError:
        return None
    return "robocasa_gr1", task
