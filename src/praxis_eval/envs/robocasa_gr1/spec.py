# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Canonical RoboCasa GR-1 observation and action schema."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

GR1_VIDEO_KEYS: tuple[str, ...] = (
    "video.ego_view_pad_res256_freq20",
    "video.ego_view_bg_crop_pad_res256_freq20",
)

# This is the concat order used by NVIDIA's FourierGr1ArmsWaist data config.
GR1_STATE_DIMS: dict[str, int] = {
    "state.left_arm": 7,
    "state.right_arm": 7,
    "state.left_hand": 6,
    "state.right_hand": 6,
    "state.waist": 3,
}
GR1_ACTION_DIMS: dict[str, int] = {
    key.replace("state.", "action."): width for key, width in GR1_STATE_DIMS.items()
}
GR1_STATE_KEYS: tuple[str, ...] = tuple(GR1_STATE_DIMS)
GR1_ACTION_KEYS: tuple[str, ...] = tuple(GR1_ACTION_DIMS)
GR1_ACTION_DIM = sum(GR1_ACTION_DIMS.values())
GR1_LANGUAGE_KEY = "annotation.human.coarse_action"


def unflatten_gr1_action(action: np.ndarray) -> dict[str, np.ndarray]:
    """Split one canonical 29-D action into the simulator's named streams."""
    action_array = np.asarray(action, dtype=np.float32)
    if action_array.shape != (GR1_ACTION_DIM,):
        raise ValueError(
            f"RoboCasa GR-1 action must have shape ({GR1_ACTION_DIM},), "
            f"got {tuple(action_array.shape)}."
        )
    if not np.isfinite(action_array).all():
        raise ValueError("RoboCasa GR-1 action contains non-finite values.")

    streams: dict[str, np.ndarray] = {}
    offset = 0
    for key, width in GR1_ACTION_DIMS.items():
        streams[key] = np.array(
            action_array[offset : offset + width], dtype=np.float32, copy=True
        )
        offset += width
    return streams


def flatten_gr1_action(action: Mapping[str, np.ndarray]) -> np.ndarray:
    """Join named GR-1 action streams in NVIDIA modality-config order."""
    missing = [key for key in GR1_ACTION_KEYS if key not in action]
    extra = sorted(set(action) - set(GR1_ACTION_KEYS))
    if missing or extra:
        raise ValueError(
            f"Invalid RoboCasa GR-1 action keys; missing={missing}, extra={extra}."
        )

    streams: list[np.ndarray] = []
    for key, width in GR1_ACTION_DIMS.items():
        value = np.asarray(action[key], dtype=np.float32)
        if value.shape != (width,):
            raise ValueError(
                f"RoboCasa GR-1 action stream {key!r} must have shape "
                f"({width},), got {tuple(value.shape)}."
            )
        streams.append(value)
    flattened = np.concatenate(streams).astype(np.float32, copy=False)
    if not np.isfinite(flattened).all():
        raise ValueError("RoboCasa GR-1 action contains non-finite values.")
    return flattened
