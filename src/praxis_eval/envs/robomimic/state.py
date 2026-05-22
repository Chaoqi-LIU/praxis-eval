# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight RoboMimic state-contract helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

ROBOMIMIC_STATE_PORTS: tuple[str, ...] = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)

STATE_PORT_SHAPES: dict[str, tuple[int, ...]] = {
    "robot0_joint_pos": (7,),
    "robot0_eef_pos": (3,),
    "robot0_eef_quat": (4,),
    "robot0_gripper_qpos": (2,),
}


def state_shape_for_port(port: str) -> tuple[int, ...]:
    """Return the tensor shape for one robosuite observation key."""
    shape = STATE_PORT_SHAPES.get(str(port))
    if shape is None:
        raise KeyError(
            f"Unknown RoboMimic state port {port!r}. "
            f"Known ports: {sorted(STATE_PORT_SHAPES)}"
        )
    return shape


def state_dim_from_ports(state_ports: Sequence[str]) -> int:
    """Return the flattened state dimension for the given ordered ports."""
    dim = 0
    for port in state_ports:
        dim += math.prod(state_shape_for_port(str(port)))
    return dim
