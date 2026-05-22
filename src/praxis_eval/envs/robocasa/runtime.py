# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa runtime helpers: dummy env for space inference and eval lane wrapper."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from praxis_eval.envs.robocasa.state import (
    ROBOCASA_STATE_PORTS,
    STATE_KEY_SHAPES,
    state_key_for_port,
)
from praxis_eval.envs.robocasa.tasks import get_subtasks, get_task_horizon

# ---------------------------------------------------------------------------
# Known constants for PandaOmron with default robocasa OSC_POSE control.
# Used by _DummyRobocasaEnv to construct spaces without instantiating the sim.
# ---------------------------------------------------------------------------

# Action dimension for PandaOmron with default composite control in this
# robocasa/robosuite stack. Must match RobocasaEnv.action_space for
# AsyncVectorEnv space validation.
_PANDA_ACTION_DIM = 12

_RETRYABLE_LAYOUT_ERROR_FRAGMENTS: tuple[str, ...] = (
    "for mesh geoms, inertia should be specified in the mesh asset",
    "mesh volume is too small",
)
_ROBOCASA_ENV_BUILD_MAX_ATTEMPTS = 8
logger = logging.getLogger(__name__)


class _DummyRobocasaEnv(gym.Env):
    """Space-inference-only env — never instantiates a real robosuite sim.

    Safe to call in the parent process before forking AsyncVectorEnv workers.
    Obs space mirrors ``RobocasaEnv``:
      ``pixels`` dict + ``robot_state`` dict (no string obs).
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        camera_names: list[str],
        image_size: int = 128,
        action_dim: int = _PANDA_ACTION_DIM,
        state_shapes: dict[str, tuple[int, ...]] | None = None,
    ):
        super().__init__()
        self.render_mode = "rgb_array"

        shapes = state_shapes or STATE_KEY_SHAPES

        pixel_spaces: dict[str, gym.Space[Any]] = {
            cam: spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            )
            for cam in camera_names
        }
        state_spaces: dict[str, gym.Space] = {}
        for port in ROBOCASA_STATE_PORTS:
            key = state_key_for_port(port)
            shape = shapes.get(key)
            if shape is None:
                raise ValueError(
                    f"Unknown state port {port!r} (key {key!r}). "
                    f"Pass state_shapes to override. Known: {list(shapes)}"
                )
            state_spaces[key] = spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
            )

        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(pixel_spaces),
                "robot_state": spaces.Dict(state_spaces),
            }
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32
        )

    def reset(self, **kwargs):
        raise NotImplementedError("_DummyRobocasaEnv is for space inference only")

    def step(self, action):
        raise NotImplementedError("_DummyRobocasaEnv is for space inference only")

    def close(self) -> None:
        pass


def make_dummy_robocasa_env_fn(
    camera_names: list[str],
    image_size: int = 128,
    action_dim: int = _PANDA_ACTION_DIM,
    state_shapes: dict[str, tuple[int, ...]] | None = None,
) -> Callable[[], _DummyRobocasaEnv]:
    """Return a no-arg callable that builds a ``_DummyRobocasaEnv``."""
    return partial(
        _DummyRobocasaEnv,
        camera_names=camera_names,
        image_size=image_size,
        action_dim=action_dim,
        state_shapes=state_shapes,
    )


def _is_retryable_layout_error(exc: Exception) -> bool:
    if not isinstance(exc, ValueError):
        return False
    return any(frag in str(exc) for frag in _RETRYABLE_LAYOUT_ERROR_FRAGMENTS)


def build_robocasa_env_with_retries(
    *,
    task_name: str,
    split: str = "all",
    image_size: int,
    seed: int,
    camera_names: list[str],
    max_episode_steps: int,
    enable_render: bool,
    max_attempts: int = _ROBOCASA_ENV_BUILD_MAX_ATTEMPTS,
) -> Any:
    """Construct RobocasaEnv with retry for known intermittent layout failures.

    Some RoboCasa scene/layout combinations crash MuJoCo XML parsing with a
    mesh-inertia ValueError. For eval-monitor robustness, retry with shifted
    seeds before failing the whole lane/job.
    """
    from praxis_eval.envs.robocasa.env import RobocasaEnv

    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}.")

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        candidate_seed = int(seed) + attempt
        try:
            return RobocasaEnv(
                task_name=task_name,
                split=cast(Literal["all", "pretrain", "target"], split),
                image_size=image_size,
                seed=candidate_seed,
                camera_names=camera_names,
                max_episode_steps=max_episode_steps,
                enable_render=enable_render,
            )
        except Exception as exc:  # pragma: no cover - exact exception type from deps
            if _is_retryable_layout_error(exc) and (attempt + 1) < max_attempts:
                logger.warning(
                    "Robocasa env build failed with retryable layout error; "
                    "task=%s seed=%d attempt=%d/%d. Retrying with seed=%d.",
                    task_name,
                    candidate_seed,
                    attempt + 1,
                    max_attempts,
                    candidate_seed + 1,
                )
                last_exc = exc
                gc.collect()
                continue
            raise

    assert last_exc is not None  # guarded by max_attempts>=1 and return/raise above
    raise RuntimeError(
        f"Failed to build RobocasaEnv for task={task_name!r} after {max_attempts} "
        f"attempts from seed={seed}."
    ) from last_exc


class RobocasaEvalLaneWrapper(gym.Wrapper):
    """Worker-local wrapper for RoboCasa eval lanes.

    Implements ``prepare_eval_job(task_id, episode_index, task_group)`` which
    switches task and per-episode reset options between eval episodes.

    Task changes rebuild the inner ``RobocasaEnv``. Same-task episodes reuse the
    existing env and only update the next reset seed/options.

    Exposes ``task_description`` as a property for ``add_envs_task``.
    """

    def __init__(
        self,
        env: Any,  # RobocasaEnv
        *,
        task_group: str,
        lane_idx: int | None = None,
    ) -> None:
        super().__init__(env)
        self._task_group = str(task_group)
        self._subtasks = get_subtasks(task_group)
        self._lane_idx = lane_idx

        # Mirror config from wrapped env so we can rebuild.
        self._split: str = str(env.split)
        self._image_size: int = env.image_size
        self._camera_names: list[str] = list(env.camera_names)
        self._default_max_episode_steps: int = int(env._max_episode_steps)

        self._current_task: str = env.task_name
        self._current_seed: int = int(getattr(env, "_seed", 0))
        self._next_reset_seed: int | None = self._current_seed
        self._next_reset_options: dict[str, Any] = {
            "episode_seed": self._current_seed,
            "reseed_inner_env": True,
        }
        self._rebuild_count = 0

    # ------------------------------------------------------------------
    # task_description / task properties (for LeRobot add_envs_task)
    # ------------------------------------------------------------------

    @property
    def task_description(self) -> str:
        return str(self.env.task_description)

    # LeRobot/gymnasium probe both `task_description` and `task` via
    # `get_wrapper_attr`. The inner RobocasaEnv has neither — expose them
    # here so the probe never walks down into an AttributeError that would
    # otherwise land in the worker error_queue and surface at the next
    # reset_wait.
    @property
    def task(self) -> str:
        return str(self.env.task_description)

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

    # ------------------------------------------------------------------
    # Eval pool protocol
    # ------------------------------------------------------------------

    def prepare_eval_job(
        self,
        task_id: int,
        episode_index: int,
        task_group: str | None = None,
        needs_rebuild: bool = False,
    ) -> None:
        """Prepare this lane for one eval episode.

        Args:
            task_id: Index into the subtask list for this group.
            episode_index: Used as the random seed for scene initialization.
            task_group: Override the task group if it changed between chunks.
        """
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild(self, *, task_name: str, seed: int) -> None:
        old_env = self.env
        # Close old env before constructing the new one: overlapping mujoco envs
        # in one process deadlock inside MjModel.from_xml_string
        # (see RobocasaEvalLaneWrapper Bug 3 stack dump 2026-04-14).
        try:
            old_env.close()
        except Exception:
            logger.warning("old env close failed; continuing", exc_info=True)
        del old_env
        gc.collect()
        new_env = build_robocasa_env_with_retries(
            task_name=task_name,
            split=self._split,
            image_size=self._image_size,
            seed=seed,
            camera_names=self._camera_names,
            max_episode_steps=get_task_horizon(
                task_name,
                default=self._default_max_episode_steps,
            ),
            enable_render=True,
        )
        self.env = new_env
        self._current_task = task_name
        self._current_seed = seed
        self._next_reset_seed = seed
        self._next_reset_options = {
            "episode_seed": seed,
            "reseed_inner_env": True,
        }
        self._rebuild_count += 1


def construct_robocasa_eval_lane(
    env_fn: Callable[[], Any],
    *,
    task_group: str,
    lane_idx: int,
) -> RobocasaEvalLaneWrapper:
    """Build one RoboCasa eval lane inside a worker process."""
    env = env_fn()
    return RobocasaEvalLaneWrapper(env, task_group=task_group, lane_idx=lane_idx)
