# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight LIBERO spaces shared by dummy and real env paths."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from praxis_eval.envs.libero.spec import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    resolve_camera_name_mapping,
)


def make_libero_action_space() -> spaces.Box:
    return spaces.Box(
        low=ACTION_LOW,
        high=ACTION_HIGH,
        shape=(ACTION_DIM,),
        dtype=np.float32,
    )


def make_libero_observation_space(
    *,
    camera_names: Sequence[str],
    obs_type: str,
    observation_height: int,
    observation_width: int,
    camera_name_mapping: dict[str, str] | None = None,
) -> spaces.Dict:
    mapping = resolve_camera_name_mapping(camera_names, camera_name_mapping)
    image_spaces: dict[str, gym.Space[Any]] = {
        mapping[camera]: spaces.Box(
            low=0,
            high=255,
            shape=(int(observation_height), int(observation_width), 3),
            dtype=np.uint8,
        )
        for camera in camera_names
    }
    pixels = spaces.Dict(cast(dict[str, spaces.Space[Any]], image_spaces))
    if obs_type == "pixels":
        return spaces.Dict({"pixels": pixels})
    if obs_type == "pixels_agent_pos":
        return spaces.Dict(
            {
                "pixels": pixels,
                "robot_state": spaces.Dict(
                    {
                        "eef": spaces.Dict(
                            {
                                "pos": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(3,),
                                    dtype=np.float64,
                                ),
                                "quat": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(4,),
                                    dtype=np.float64,
                                ),
                                "mat": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(3, 3),
                                    dtype=np.float64,
                                ),
                            }
                        ),
                        "gripper": spaces.Dict(
                            {
                                "qpos": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(2,),
                                    dtype=np.float64,
                                ),
                                "qvel": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(2,),
                                    dtype=np.float64,
                                ),
                            }
                        ),
                        "joints": spaces.Dict(
                            {
                                "pos": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(7,),
                                    dtype=np.float64,
                                ),
                                "vel": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(7,),
                                    dtype=np.float64,
                                ),
                            }
                        ),
                    }
                ),
            }
        )
    raise ValueError(f"Unsupported LIBERO obs_type: {obs_type!r}.")
