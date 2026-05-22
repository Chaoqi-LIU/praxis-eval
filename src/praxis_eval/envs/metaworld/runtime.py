# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""MetaWorld runtime helpers for async eval lanes."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import gymnasium as gym

from praxis_eval.envs.metaworld.env import (
    MetaworldEnv,
    make_action_space,
    make_observation_space,
)
from praxis_eval.envs.metaworld.spec import DEFAULT_MAX_EPISODE_STEPS
from praxis_eval.envs.metaworld.tasks import get_task_description, resolve_task_name


class _DummyMetaworldEnv(gym.Env):
    """Space-inference-only env that never instantiates MuJoCo."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        *,
        obs_type: str = "pixels_agent_pos",
        observation_height: int = 480,
        observation_width: int = 480,
    ) -> None:
        super().__init__()
        self.render_mode = "rgb_array"
        self.observation_space = make_observation_space(
            obs_type=obs_type,
            observation_height=observation_height,
            observation_width=observation_width,
        )
        self.action_space = make_action_space()

    def reset(self, **kwargs):
        raise NotImplementedError("_DummyMetaworldEnv is for space inference only.")

    def step(self, action):
        raise NotImplementedError("_DummyMetaworldEnv is for space inference only.")

    def close(self) -> None:
        pass


def make_dummy_metaworld_env_fn(
    *,
    obs_type: str = "pixels_agent_pos",
    observation_height: int = 480,
    observation_width: int = 480,
) -> Callable[[], _DummyMetaworldEnv]:
    """Return a no-arg callable that builds a dummy MetaWorld env."""
    return partial(
        _DummyMetaworldEnv,
        obs_type=obs_type,
        observation_height=observation_height,
        observation_width=observation_width,
    )


def make_metaworld_env_fn(
    *,
    task_name: str,
    camera_name: str = "corner2",
    obs_type: str = "pixels_agent_pos",
    render_mode: str = "rgb_array",
    observation_width: int = 480,
    observation_height: int = 480,
    visualization_width: int = 640,
    visualization_height: int = 480,
    episode_length: int | None = None,
) -> Callable[[], Any]:
    """Return a real MetaWorld env factory for one eval lane assignment."""
    return partial(
        MetaworldEnv,
        task=task_name,
        camera_name=camera_name,
        obs_type=obs_type,
        render_mode=render_mode,
        observation_width=int(observation_width),
        observation_height=int(observation_height),
        visualization_width=int(visualization_width),
        visualization_height=int(visualization_height),
        episode_length=episode_length,
    )


def construct_metaworld_eval_lane(
    env_fn: Callable[[], Any],
    *,
    task_group: str,
    lane_idx: int,
) -> MetaworldEvalLaneWrapper:
    """Build one MetaWorld eval lane."""
    return MetaworldEvalLaneWrapper(
        env_fn(),
        task_group=task_group,
        lane_idx=lane_idx,
    )


class MetaworldEvalLaneWrapper(gym.Wrapper):
    """Worker-local wrapper for MetaWorld eval lane retargeting."""

    def __init__(
        self,
        env: Any,
        *,
        task_group: str,
        lane_idx: int | None = None,
    ) -> None:
        super().__init__(env)
        self._task_group = str(task_group)
        self._lane_idx = lane_idx

        self._camera_name = str(env.camera_name)
        self._obs_type = str(env.obs_type)
        self._render_mode = str(env.render_mode or "rgb_array")
        self._observation_width = int(env.observation_width)
        self._observation_height = int(env.observation_height)
        self._visualization_width = int(env.visualization_width)
        self._visualization_height = int(env.visualization_height)
        self._episode_length = getattr(env, "_max_episode_steps", None)
        self._current_task_name = str(env.task)

    @property
    def task(self) -> str:
        return str(getattr(self.env, "task", self._current_task_name))

    @property
    def task_description(self) -> str:
        description = getattr(self.env, "task_description", None)
        if description is not None:
            return str(description)
        return get_task_description(self.task)

    @property
    def _max_episode_steps(self) -> int:
        value = getattr(self.env, "_max_episode_steps", self._episode_length)
        return int(value) if value is not None else DEFAULT_MAX_EPISODE_STEPS

    def prepare_eval_job(
        self,
        task_id: int,
        episode_index: int,
        task_group: str | None = None,
    ) -> None:
        """Prepare this lane for one eval job."""
        _ = episode_index
        if task_group is not None:
            self._task_group = str(task_group)
        task_name = resolve_task_name(self._task_group, int(task_id))
        if task_name != self._current_task_name:
            self._rebuild_env(task_name=task_name)

    def _rebuild_env(self, *, task_name: str) -> None:
        self.env.close()
        self.env = MetaworldEnv(
            task=task_name,
            camera_name=self._camera_name,
            obs_type=self._obs_type,
            render_mode=self._render_mode,
            observation_width=self._observation_width,
            observation_height=self._observation_height,
            visualization_width=self._visualization_width,
            visualization_height=self._visualization_height,
            episode_length=self._episode_length,
        )
        self._current_task_name = task_name
