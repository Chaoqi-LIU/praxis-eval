# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight LIBERO constants and selectors."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
NOOP_ACTION = (0, 0, 0, 0, 0, 0, -1)
DEFAULT_CAMERA_NAME = "agentview_image,robot0_eye_in_hand_image"
DEFAULT_CAMERA_NAME_MAPPING = {
    "agentview_image": "image",
    "robot0_eye_in_hand_image": "image2",
}
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize a LIBERO camera selector into a non-empty camera list."""
    if isinstance(camera_name, str):
        names = [part.strip() for part in camera_name.split(",") if part.strip()]
    elif isinstance(camera_name, (list | tuple)):
        names = [str(part).strip() for part in camera_name if str(part).strip()]
    else:
        raise TypeError(
            "camera_name must be a comma-separated string or a sequence of strings, "
            f"got {type(camera_name).__name__}."
        )
    if not names:
        raise ValueError("camera_name resolved to an empty camera list.")
    return names


def resolve_camera_name_mapping(
    camera_names: Sequence[str],
    camera_name_mapping: dict[str, str] | None,
) -> dict[str, str]:
    mapping = dict(DEFAULT_CAMERA_NAME_MAPPING)
    if camera_name_mapping is not None:
        mapping.update(
            {str(key): str(value) for key, value in camera_name_mapping.items()}
        )

    missing = [camera for camera in camera_names if camera not in mapping]
    if missing:
        raise ValueError(
            "Missing LIBERO camera_name_mapping entries for cameras: "
            f"{', '.join(missing)}."
        )
    return {camera: mapping[camera] for camera in camera_names}


def select_task_ids(
    total_tasks: int,
    task_ids: Iterable[int] | None,
) -> list[int]:
    """Validate and normalize task ids; ``None`` selects every task."""
    if task_ids is None:
        return list(range(total_tasks))

    ids = sorted({int(task_id) for task_id in task_ids})
    for task_id in ids:
        if task_id < 0 or task_id >= total_tasks:
            raise ValueError(f"task_id {task_id} out of range [0, {total_tasks - 1}].")
    return ids
