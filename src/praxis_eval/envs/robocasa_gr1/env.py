# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""In-process Praxis adapter for RoboCasa GR-1 Gym environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from praxis_eval.envs.robocasa_gr1.spec import (
    GR1_ACTION_DIM,
    GR1_LANGUAGE_KEY,
    GR1_STATE_DIMS,
    GR1_VIDEO_KEYS,
    unflatten_gr1_action,
)
from praxis_eval.envs.robocasa_gr1.tasks import resolve_gr1_task


class RobocasaGr1Env(gym.Env):
    """Flattened-action wrapper around the official GR-1 GR00T environment."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task_name: str,
        *,
        max_episode_steps: int = 720,
        enable_render: bool = True,
    ) -> None:
        super().__init__()
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1.")

        # Importing this module creates the official dynamic Gym registrations.
        import robocasa_gr1.utils.gym_utils.gymnasium_groot  # noqa: F401

        self.task_name = resolve_gr1_task(task_name)
        self._max_episode_steps = int(max_episode_steps)
        self.enable_render = bool(enable_render)
        self.env = gym.make(
            self.task_name,
            enable_render=self.enable_render,
            disable_env_checker=True,
        )
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(GR1_ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = self._build_observation_space()
        self._step_count = 0
        self._task_description = self.task_name
        self._render_cache: np.ndarray | None = None
        self._done = False

    @property
    def task_description(self) -> str:
        return self._task_description

    @property
    def task(self) -> str:
        return self._task_description

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_obs, info = self.env.reset(seed=seed, options=options)
        self._step_count = 0
        self._done = False
        self._task_description = str(raw_obs.get(GR1_LANGUAGE_KEY, self.task_name))
        return self._adapt_observation(raw_obs), dict(info)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action_dict = unflatten_gr1_action(action)
        raw_obs, reward, terminated, truncated, info = self.env.step(action_dict)
        self._step_count += 1
        info = dict(info)
        is_success = bool(info.get("success", reward > 0))
        info["is_success"] = is_success
        terminated_out = bool(terminated or is_success)
        truncated_out = bool(
            truncated
            or (self._step_count >= self._max_episode_steps and not terminated_out)
        )
        self._done = bool(self._done or terminated_out or truncated_out)
        if self._done:
            info["final_info"] = {
                "task": self.task_name,
                "done": self._done,
                "is_success": is_success,
            }
        return (
            self._adapt_observation(raw_obs),
            float(reward),
            terminated_out,
            truncated_out,
            info,
        )

    def render(self) -> np.ndarray:
        if self._render_cache is None:
            raise RuntimeError("Must call reset() before render().")
        return self._render_cache

    def close(self) -> None:
        self.env.close()

    def _build_observation_space(self) -> spaces.Dict:
        pixel_spaces = {
            key: spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8)
            for key in GR1_VIDEO_KEYS
        }
        state_spaces = {
            key: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(width,),
                dtype=np.float32,
            )
            for key, width in GR1_STATE_DIMS.items()
        }
        return spaces.Dict(
            {
                "pixels": spaces.Dict(pixel_spaces),
                "robot_state": spaces.Dict(state_spaces),
            }
        )

    def _adapt_observation(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        pixels: dict[str, np.ndarray] = {}
        for key in GR1_VIDEO_KEYS:
            image = np.asarray(raw_obs[key], dtype=np.uint8)
            if image.shape != (256, 256, 3):
                raise ValueError(
                    f"RoboCasa GR-1 image {key!r} has shape {image.shape}, "
                    "expected (256, 256, 3)."
                )
            pixels[key] = image

        state: dict[str, np.ndarray] = {}
        for key, width in GR1_STATE_DIMS.items():
            value = np.asarray(raw_obs[key], dtype=np.float32)
            if value.shape != (width,):
                raise ValueError(
                    f"RoboCasa GR-1 state {key!r} has shape {value.shape}, "
                    f"expected ({width},)."
                )
            state[key] = value

        self._render_cache = pixels[GR1_VIDEO_KEYS[0]]
        return {"pixels": pixels, "robot_state": state}
