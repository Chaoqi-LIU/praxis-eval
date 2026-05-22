# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight RoboCasa state-contract helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

# Canonical RoboCasa v1.0 / LeRobot proprio contract used by the official
# RoboCasa datasets and the live mt5 merged dataset.
ROBOCASA_STATE_PORTS: tuple[str, ...] = (
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_gripper_qpos",
)

STATE_KEY_SHAPES: dict[str, tuple[int, ...]] = {
    "base_pos": (3,),
    "base_quat": (4,),
    "base_to_eef_pos": (3,),
    "base_to_eef_quat": (4,),
    "gripper_qpos": (2,),
    # Optional alternate / legacy observation keys that evaluator tests and
    # experiments may still request explicitly.
    "joint_pos": (7,),
    "joint_vel": (7,),
    "gripper_qvel": (2,),
    "eef_pos": (3,),
    "eef_quat": (4,),
}


def state_key_for_port(port: str) -> str:
    """Strip the RoboCasa robot prefix from a state observation key."""
    return str(port).removeprefix("robot0_")


def state_keys_from_ports(state_ports: Sequence[str]) -> tuple[str, ...]:
    """Convert robosuite-style state ports to flattened state-key names."""
    return tuple(state_key_for_port(port) for port in state_ports)


def state_shape_for_port(port: str) -> tuple[int, ...]:
    """Return the tensor shape for one robosuite-style state port."""
    key = state_key_for_port(port)
    shape = STATE_KEY_SHAPES.get(key)
    if shape is None:
        raise KeyError(
            f"Unknown RoboCasa state port {port!r} (key {key!r}). "
            f"Known keys: {sorted(STATE_KEY_SHAPES)}"
        )
    return shape


def state_dim_from_keys(state_keys: Sequence[str]) -> int:
    """Return the flattened state dimension for the given ordered keys."""
    dim = 0
    for key in state_keys:
        shape = STATE_KEY_SHAPES.get(str(key))
        if shape is None:
            raise KeyError(
                f"Unknown RoboCasa state key {key!r}. Known: {sorted(STATE_KEY_SHAPES)}"
            )
        dim += math.prod(shape)
    return dim
