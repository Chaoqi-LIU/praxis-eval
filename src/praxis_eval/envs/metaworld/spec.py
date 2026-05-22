# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared MetaWorld environment contract constants."""

from __future__ import annotations

from numbers import Integral

ACTION_DIM = 4
OBS_DIM = 4
DEFAULT_MAX_EPISODE_STEPS = 500
LEROBOT_CORNER2_CAMERA_POSITION = [0.75, 0.075, 0.7]


def resolve_episode_length(episode_length: int | None) -> int:
    """Return the effective MetaWorld horizon, rejecting invalid configs early."""
    if episode_length is None:
        return DEFAULT_MAX_EPISODE_STEPS
    if isinstance(episode_length, bool) or not isinstance(episode_length, Integral):
        raise ValueError("MetaWorld episode_length must be a positive integer or None.")
    value = int(episode_length)
    if value <= 0:
        raise ValueError("MetaWorld episode_length must be a positive integer or None.")
    return value
