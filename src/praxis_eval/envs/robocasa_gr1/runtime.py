# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Worker-local runtime helpers for RoboCasa GR-1 evaluation."""

from __future__ import annotations

import gc
from collections.abc import Callable
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from praxis_eval.envs.robocasa_gr1.env import RobocasaGr1Env
from praxis_eval.envs.robocasa_gr1.spec import (
    GR1_ACTION_DIM,
    GR1_STATE_DIMS,
    GR1_VIDEO_KEYS,
)


class _DummyRobocasaGr1Env(gym.Env):
    """Space-only env used before async workers import MuJoCo/OpenGL."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape=(GR1_ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        key: spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8)
                        for key in GR1_VIDEO_KEYS
                    }
                ),
                "robot_state": spaces.Dict(
                    {
                        key: spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(width,),
                            dtype=np.float32,
                        )
                        for key, width in GR1_STATE_DIMS.items()
                    }
                ),
            }
        )

    def reset(self, **kwargs):
        raise NotImplementedError("Dummy GR-1 env is only for space inference.")

    def step(self, action):
        raise NotImplementedError("Dummy GR-1 env is only for space inference.")


def make_dummy_robocasa_gr1_env_fn() -> Callable[[], _DummyRobocasaGr1Env]:
    return partial(_DummyRobocasaGr1Env)


class RobocasaGr1EvalLaneWrapper(gym.Wrapper):
    """Persistent eval lane that rebuilds only when its GR-1 task changes."""

    def __init__(
        self,
        env: RobocasaGr1Env,
        *,
        lane_idx: int | None = None,
        initial_seed: int = 0,
    ) -> None:
        super().__init__(env)
        self._lane_idx = lane_idx
        self._current_task = env.task_name
        self._max_steps = int(env._max_episode_steps)
        self._next_seed: int | None = int(initial_seed)

    @property
    def task_description(self) -> str:
        return str(self.env.task_description)

    @property
    def task(self) -> str:
        return str(self.env.task)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Any:
        reset_seed = self._next_seed if seed is None else int(seed)
        self._next_seed = None
        return self.env.reset(seed=reset_seed, options=options)

    def prepare_eval_job(
        self,
        task_id: int,
        episode_index: int,
        task_group: str | None = None,
        needs_rebuild: bool = False,
    ) -> None:
        _ = task_id
        task_name = self._current_task if task_group is None else str(task_group)
        seed = int(episode_index)
        if needs_rebuild or task_name != self._current_task:
            old_env = self.env
            old_env.close()
            del old_env
            gc.collect()
            self.env = RobocasaGr1Env(
                task_name,
                max_episode_steps=self._max_steps,
                enable_render=True,
            )
            self._current_task = task_name
        self._next_seed = seed


def construct_robocasa_gr1_eval_lane(
    env_fn: Callable[[], RobocasaGr1Env],
    *,
    lane_idx: int,
    initial_seed: int = 0,
) -> RobocasaGr1EvalLaneWrapper:
    return RobocasaGr1EvalLaneWrapper(
        env_fn(), lane_idx=lane_idx, initial_seed=initial_seed
    )
