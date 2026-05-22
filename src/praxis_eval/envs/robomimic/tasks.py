# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboMimic task definitions and eval-target inference."""

from __future__ import annotations

from typing import Any

LEAF_TASKS: tuple[str, ...] = (
    "Lift",
    "PickPlaceCan",
    "NutAssemblySquare",
    "ToolHang",
)

MT_TASKS: dict[str, list[str]] = {
    "mt3": [
        "Lift",
        "PickPlaceCan",
        "NutAssemblySquare",
    ],
    "mt4": [
        "Lift",
        "PickPlaceCan",
        "ToolHang",
        "NutAssemblySquare",
    ],
}

TASK_INSTRUCTIONS: dict[str, str] = {
    "Lift": "Lift the cube.",
    "PickPlaceCan": "Pick up the can and place it in the target bin.",
    "NutAssemblySquare": "Fit the square nut onto its matching peg.",
    "ToolHang": "Assemble the stand and hang the wrench on it.",
}

# Rounded-up caps over the cached RoboMimic v1.5 PH mt4 LeRobot dataset
# (`chaoqi-liu/robomimic_mt4_ph`). The env config default remains a fallback for
# unknown tasks, while eval lanes use these leaf-task caps.
TASK_HORIZONS: dict[str, int] = {
    "Lift": 100,
    "PickPlaceCan": 200,
    "NutAssemblySquare": 300,
    "ToolHang": 800,
}

_TASK_ALIASES: dict[str, str] = {
    "lift": "Lift",
    "can": "PickPlaceCan",
    "pickplacecan": "PickPlaceCan",
    "pick_place_can": "PickPlaceCan",
    "square": "NutAssemblySquare",
    "nutassemblysquare": "NutAssemblySquare",
    "nut_assembly_square": "NutAssemblySquare",
    "toolhang": "ToolHang",
    "tool_hang": "ToolHang",
}


def canonicalize_task_name(task_name: str) -> str:
    """Return canonical robosuite env name for known RoboMimic aliases."""
    task = str(task_name).strip()
    if task in MT_TASKS or task in LEAF_TASKS:
        return task
    return _TASK_ALIASES.get(task.lower(), task)


def is_multitask(task_name: str) -> bool:
    """Return True if task_name refers to a evaluator RoboMimic task group."""
    return canonicalize_task_name(task_name) in MT_TASKS


def get_subtasks(task_name: str) -> list[str]:
    """Expand a RoboMimic task name or group into canonical leaf tasks."""
    canonical = canonicalize_task_name(task_name)
    if canonical in MT_TASKS:
        return list(MT_TASKS[canonical])
    return [canonical]


def get_task_horizon(task_name: str, *, default: int = 800) -> int:
    """Return the rollout horizon for a RoboMimic task or group."""
    subtasks = get_subtasks(task_name)
    if not subtasks:
        return int(default)
    return max(TASK_HORIZONS.get(subtask, int(default)) for subtask in subtasks)


def get_task_instruction(task_name: str) -> str:
    """Return the natural-language policy instruction for a RoboMimic task."""
    canonical = canonicalize_task_name(task_name)
    return TASK_INSTRUCTIONS.get(canonical, canonical)


def list_robomimic_tasks(
    raw_cfg: dict[str, Any],
    cfg_obj: Any,
    debug_verbose: bool = False,
) -> list[tuple[str, int]]:
    """List ``(task_name, task_id)`` pairs for the configured RoboMimic target."""
    _ = debug_verbose
    task = canonicalize_task_name(
        str(getattr(cfg_obj, "task", raw_cfg.get("task", "mt3")))
    )
    subtasks = get_subtasks(task)
    return [(subtask, idx) for idx, subtask in enumerate(subtasks)]


def infer_robomimic_eval_target_from_dataset(
    dataset_name: str,
) -> tuple[str, str] | None:
    """Infer ``(env.type, env.task)`` from a RoboMimic dataset name."""
    prefix = "robomimic_"
    if not dataset_name.startswith(prefix):
        return None
    task = dataset_name[len(prefix) :]
    if task.endswith("_ph"):
        task = task[: -len("_ph")]
    task = canonicalize_task_name(task)
    if task in MT_TASKS or task in LEAF_TASKS:
        return "robomimic", task
    return None
