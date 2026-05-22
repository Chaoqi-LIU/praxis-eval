# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboMimic runtime helpers for async eval lanes."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from praxis_eval.envs.robomimic.state import ROBOMIMIC_STATE_PORTS, state_shape_for_port
from praxis_eval.envs.robomimic.tasks import get_subtasks, get_task_horizon

_PANDA_ACTION_DIM = 7
_ROBOMIMIC_ENV_BUILD_MAX_ATTEMPTS = 1
logger = logging.getLogger(__name__)


class _DummyRobomimicEnv(gym.Env):
    """Space-inference-only env that never instantiates robosuite."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        camera_names: list[str],
        image_size: int = 128,
        state_ports: list[str] | None = None,
        action_dim: int = _PANDA_ACTION_DIM,
    ) -> None:
        super().__init__()
        self.render_mode = "rgb_array"
        ports = list(state_ports or ROBOMIMIC_STATE_PORTS)
        pixel_spaces: dict[str, gym.Space[Any]] = {
            cam: spaces.Box(
                low=0,
                high=255,
                shape=(image_size, image_size, 3),
                dtype=np.uint8,
            )
            for cam in camera_names
        }
        state_spaces: dict[str, gym.Space[Any]] = {
            port: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=state_shape_for_port(port),
                dtype=np.float32,
            )
            for port in ports
        }
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(pixel_spaces),
                "robot_state": spaces.Dict(state_spaces),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(action_dim),),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        raise NotImplementedError("_DummyRobomimicEnv is for space inference only")

    def step(self, action):
        raise NotImplementedError("_DummyRobomimicEnv is for space inference only")

    def close(self) -> None:
        pass


def make_dummy_robomimic_env_fn(
    camera_names: list[str],
    image_size: int = 128,
    state_ports: list[str] | None = None,
    action_dim: int = _PANDA_ACTION_DIM,
) -> Callable[[], _DummyRobomimicEnv]:
    """Return a no-arg callable that builds a dummy RoboMimic env."""
    return partial(
        _DummyRobomimicEnv,
        camera_names=camera_names,
        image_size=image_size,
        state_ports=state_ports,
        action_dim=action_dim,
    )


def build_robomimic_env_with_retries(
    *,
    task_name: str,
    image_size: int,
    seed: int,
    camera_names: list[str],
    state_ports: list[str],
    video_camera: str,
    video_resolution: int,
    max_episode_steps: int,
    enable_render: bool,
    robot: str,
    max_attempts: int = _ROBOMIMIC_ENV_BUILD_MAX_ATTEMPTS,
) -> Any:
    """Construct ``RobomimicEnv`` with a narrow retry hook."""
    from praxis_eval.envs.robomimic.env import RobomimicEnv

    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}.")

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        candidate_seed = int(seed) + attempt
        try:
            return RobomimicEnv(
                task_name=task_name,
                image_size=image_size,
                seed=candidate_seed,
                camera_names=camera_names,
                state_ports=state_ports,
                video_camera=video_camera,
                video_resolution=video_resolution,
                max_episode_steps=max_episode_steps,
                enable_render=enable_render,
                robot=robot,
            )
        except Exception as exc:  # pragma: no cover - exact type comes from robosuite
            last_exc = exc
            if attempt + 1 >= max_attempts:
                break
            logger.warning(
                "RoboMimic env build failed; task=%s seed=%d attempt=%d/%d. "
                "Retrying with seed=%d.",
                task_name,
                candidate_seed,
                attempt + 1,
                max_attempts,
                candidate_seed + 1,
                exc_info=True,
            )
            gc.collect()

    assert last_exc is not None
    raise RuntimeError(
        f"Failed to build RobomimicEnv for task={task_name!r} after {max_attempts} "
        f"attempts from seed={seed}."
    ) from last_exc


class RobomimicEvalLaneWrapper(gym.Wrapper):
    """Worker-local wrapper for RoboMimic eval lane retargeting."""

    def __init__(
        self,
        env: Any,
        *,
        task_group: str,
        lane_idx: int | None = None,
    ) -> None:
        super().__init__(env)
        self._task_group = str(task_group)
        self._subtasks = get_subtasks(task_group)
        self._lane_idx = lane_idx

        self._image_size = int(env.image_size)
        self._camera_names = list(env.camera_names)
        self._state_ports = list(env.state_ports)
        self._video_camera = str(env.video_camera)
        self._video_resolution = int(env.video_resolution)
        self._default_max_episode_steps = int(env._max_episode_steps)
        self._robot = str(env.robot)

        self._current_task = str(env.task_name)
        self._current_seed = int(getattr(env, "_seed", 0))
        self._next_reset_seed: int | None = self._current_seed
        self._next_reset_options: dict[str, Any] = {
            "episode_seed": self._current_seed,
            "reseed_inner_env": True,
        }
        self._rebuild_count = 0

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
        if seed is None:
            reset_seed = self._next_reset_seed
            reset_options = dict(self._next_reset_options)
            if options is not None:
                reset_options.update(options)
        else:
            reset_seed = int(seed)
            reset_options = {
                "episode_seed": int(seed),
                "reseed_inner_env": True,
            }
            if options is not None:
                reset_options.update(options)

        self._current_seed = int(reset_options.get("episode_seed", self._current_seed))
        self._next_reset_seed = None
        self._next_reset_options = {
            "episode_seed": self._current_seed,
            "reseed_inner_env": False,
        }
        return self.env.reset(seed=reset_seed, options=reset_options)

    def prepare_eval_job(
        self,
        task_id: int,
        episode_index: int,
        task_group: str | None = None,
        needs_rebuild: bool = False,
    ) -> None:
        """Prepare this lane for one eval episode."""
        if task_group is not None and task_group != self._task_group:
            self._task_group = str(task_group)
            self._subtasks = get_subtasks(task_group)

        task_name = self._subtasks[int(task_id) % len(self._subtasks)]
        seed = int(episode_index)
        needs_rebuild = needs_rebuild or task_name != self._current_task

        if needs_rebuild:
            self._rebuild(task_name=task_name, seed=seed)
            return

        self._next_reset_seed = None
        self._next_reset_options = {
            "episode_seed": seed,
            "reseed_inner_env": False,
        }

    def _rebuild(self, *, task_name: str, seed: int) -> None:
        old_env = self.env
        try:
            old_env.close()
        except Exception:
            logger.warning("old RoboMimic env close failed; continuing", exc_info=True)
        del old_env
        gc.collect()

        max_episode_steps = get_task_horizon(
            task_name,
            default=self._default_max_episode_steps,
        )
        self.env = build_robomimic_env_with_retries(
            task_name=task_name,
            image_size=self._image_size,
            seed=seed,
            camera_names=self._camera_names,
            state_ports=self._state_ports,
            video_camera=self._video_camera,
            video_resolution=self._video_resolution,
            max_episode_steps=max_episode_steps,
            enable_render=True,
            robot=self._robot,
        )
        self._current_task = task_name
        self._current_seed = seed
        self._next_reset_seed = seed
        self._next_reset_options = {
            "episode_seed": seed,
            "reseed_inner_env": True,
        }
        self._rebuild_count += 1


def construct_robomimic_eval_lane(
    env_fn: Callable[[], Any],
    *,
    task_group: str,
    lane_idx: int,
) -> RobomimicEvalLaneWrapper:
    """Build one RoboMimic eval lane inside a worker process."""
    env = env_fn()
    return RobomimicEvalLaneWrapper(env, task_group=task_group, lane_idx=lane_idx)
