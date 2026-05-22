# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO env runtime helpers for async eval and parent-side space bootstrap."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import gymnasium as gym

from praxis_eval.envs.libero.output import suppress_libero_output
from praxis_eval.envs.libero.spaces import (
    make_libero_action_space,
    make_libero_observation_space,
)
from praxis_eval.envs.libero.spec import ACTION_DIM, NOOP_ACTION

if TYPE_CHECKING:
    from praxis_eval.envs.libero.env import LiberoEnv


def construct_libero_eval_lane(
    env_fn: Callable[[], LiberoEnv],
    *,
    suite_name: str,
    lane_idx: int,
    debug_verbose: bool,
) -> LiberoEvalLaneWrapper:
    """Build one LIBERO eval lane, optionally suppressing noisy upstream output."""
    with suppress_libero_output(not debug_verbose):
        env = env_fn()
    return LiberoEvalLaneWrapper(
        env,
        suite_name=suite_name,
        lane_idx=lane_idx,
        debug_verbose=debug_verbose,
    )


class _DummyLiberoEnv(gym.Env):
    """Lightweight env exposing LIBERO-compatible observation/action spaces."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        camera_names: list[str],
        obs_type: str = "pixels_agent_pos",
        observation_height: int = 256,
        observation_width: int = 256,
        camera_name_mapping: dict[str, str] | None = None,
    ):
        super().__init__()
        self.render_mode = "rgb_array"

        self.observation_space = make_libero_observation_space(
            camera_names=camera_names,
            obs_type=obs_type,
            observation_height=observation_height,
            observation_width=observation_width,
            camera_name_mapping=camera_name_mapping,
        )
        self.action_space = make_libero_action_space()

    def reset(self, **kwargs):
        raise NotImplementedError("_DummyLiberoEnv is for space inference only")

    def step(self, action):
        raise NotImplementedError("_DummyLiberoEnv is for space inference only")

    def close(self):
        pass


def make_dummy_libero_env_fn(
    camera_names: list[str],
    obs_type: str = "pixels_agent_pos",
    observation_height: int = 256,
    observation_width: int = 256,
    camera_name_mapping: dict[str, str] | None = None,
) -> Callable[[], _DummyLiberoEnv]:
    """Return a no-arg callable that constructs a `_DummyLiberoEnv`."""
    return partial(
        _DummyLiberoEnv,
        camera_names=camera_names,
        obs_type=obs_type,
        observation_height=observation_height,
        observation_width=observation_width,
        camera_name_mapping=camera_name_mapping,
    )


def make_libero_env_fn(
    *,
    suite: Any,
    suite_name: str,
    task_id: int,
    episode_index: int,
    reset_stride: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: dict[str, Any],
    control_mode: str,
) -> Callable[[], LiberoEnv]:
    """Return a real LIBERO env factory for one eval lane assignment."""
    from praxis_eval.envs.libero.env import LiberoEnv

    return partial(
        LiberoEnv,
        task_suite=suite,
        task_id=int(task_id),
        task_suite_name=str(suite_name),
        episode_length=episode_length,
        camera_name=list(camera_names),
        init_states=bool(init_states),
        episode_index=int(episode_index),
        n_envs=int(reset_stride),
        control_mode=str(control_mode),
        **dict(gym_kwargs),
    )


class LiberoEvalLaneWrapper(gym.Wrapper):
    """Worker-local control wrapper for LIBERO eval lanes."""

    def __init__(
        self,
        env: LiberoEnv,
        *,
        suite_name: str,
        lane_idx: int | None = None,
        debug_verbose: bool = False,
    ) -> None:
        super().__init__(env)
        self._suite_name = str(suite_name)
        self._suite = None
        self._lane_idx = None if lane_idx is None else int(lane_idx)
        self._debug_verbose = bool(debug_verbose)

        self._episode_length = env.episode_length
        self._camera_name = list(env.camera_name)
        self._obs_type = env.obs_type
        self._render_mode = env.render_mode or "rgb_array"
        self._observation_width = int(env.observation_width)
        self._observation_height = int(env.observation_height)
        self._visualization_width = int(env.visualization_width)
        self._visualization_height = int(env.visualization_height)
        self._init_states = bool(env.init_states)
        self._n_envs = int(getattr(env, "_reset_stride", 1))
        self._camera_name_mapping = dict(env.camera_name_mapping)
        self._num_steps_wait = int(env.num_steps_wait)
        self._control_mode = str(env.control_mode)

        noop_len = len(NOOP_ACTION)
        if noop_len != ACTION_DIM:
            raise ValueError(
                f"Unexpected LIBERO dummy action length: expected {ACTION_DIM}, got {noop_len}."
            )

    @property
    def task(self) -> str:
        return str(getattr(self.env, "task", ""))

    @property
    def task_description(self) -> str:
        return str(getattr(self.env, "task_description", ""))

    def prepare_eval_job(
        self, task_id: int, episode_index: int, task_group: str | None = None
    ) -> None:
        """Prepare this lane for one eval job."""
        task_id = int(task_id)
        episode_index = int(episode_index)

        suite_changed = False
        if task_group is not None and task_group != self._suite_name:
            self._suite_name = str(task_group)
            self._suite = None
            suite_changed = True

        current_task_id = int(getattr(self.env, "task_id", -1))
        if task_id != current_task_id or suite_changed:
            self._rebuild_env(task_id=task_id, episode_index=episode_index)
        else:
            self.env.init_state_id = episode_index

    def _rebuild_env(self, *, task_id: int, episode_index: int) -> None:
        from praxis_eval.envs.libero.env import LiberoEnv, get_suite

        self.env.close()
        with suppress_libero_output(not self._debug_verbose):
            if self._suite is None:
                self._suite = get_suite(self._suite_name)

            self.env = LiberoEnv(
                task_suite=self._suite,
                task_id=int(task_id),
                task_suite_name=self._suite_name,
                episode_length=self._episode_length,
                camera_name=self._camera_name,
                obs_type=self._obs_type,
                render_mode=self._render_mode,
                observation_width=self._observation_width,
                observation_height=self._observation_height,
                visualization_width=self._visualization_width,
                visualization_height=self._visualization_height,
                init_states=self._init_states,
                episode_index=int(episode_index),
                n_envs=self._n_envs,
                camera_name_mapping=self._camera_name_mapping,
                num_steps_wait=self._num_steps_wait,
                control_mode=self._control_mode,
            )
