# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""evaluator-owned MetaWorld Gymnasium environment."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from praxis_eval.envs.metaworld._compat import patch_metaworld_env
from praxis_eval.envs.metaworld.spec import (
    ACTION_DIM,
    LEROBOT_CORNER2_CAMERA_POSITION,
    OBS_DIM,
    resolve_episode_length,
)
from praxis_eval.envs.metaworld.tasks import (
    canonicalize_task_selector,
    get_task_description,
)


def make_observation_space(
    *,
    obs_type: str,
    observation_height: int,
    observation_width: int,
) -> spaces.Dict:
    pixel_space = spaces.Box(
        low=0,
        high=255,
        shape=(int(observation_height), int(observation_width), 3),
        dtype=np.uint8,
    )
    if obs_type == "pixels":
        return spaces.Dict({"pixels": pixel_space})
    if obs_type == "pixels_agent_pos":
        return spaces.Dict(
            {
                "pixels": pixel_space,
                "agent_pos": spaces.Box(
                    low=-1000.0,
                    high=1000.0,
                    shape=(OBS_DIM,),
                    dtype=np.float64,
                ),
            }
        )
    raise ValueError(f"Unsupported MetaWorld obs_type: {obs_type!r}.")


def make_action_space() -> spaces.Box:
    return spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(ACTION_DIM,),
        dtype=np.float32,
    )


class MetaworldEnv(gym.Env):
    """Minimal MetaWorld wrapper with the observation contract Praxis policies use."""

    metadata: dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 80,
    }

    def __init__(
        self,
        *,
        task: str,
        camera_name: str = "corner2",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 480,
        observation_height: int = 480,
        visualization_width: int = 640,
        visualization_height: int = 480,
        episode_length: int | None = None,
    ) -> None:
        super().__init__()
        self.task = canonicalize_task_selector(task)
        self.camera_name = str(camera_name)
        self.obs_type = str(obs_type)
        self.render_mode = str(render_mode)
        self.observation_width = int(observation_width)
        self.observation_height = int(observation_height)
        self.visualization_width = int(visualization_width)
        self.visualization_height = int(visualization_height)
        self._max_episode_steps = resolve_episode_length(episode_length)
        self.task_description = get_task_description(self.task)

        self.observation_space = make_observation_space(
            obs_type=self.obs_type,
            observation_height=self.observation_height,
            observation_width=self.observation_width,
        )
        self.action_space = make_action_space()
        self._env: Any | None = None
        self._step_count = 0

    def _ensure_env(self) -> Any:
        if self._env is None:
            self._env = self._make_backend_env()
        return self._env

    def _make_backend_env(self) -> Any:
        patch_metaworld_env()
        import metaworld

        mt1 = metaworld.MT1(self.task, seed=42)
        env = mt1.train_classes[self.task](
            render_mode="rgb_array",
            camera_name=self.camera_name,
            width=self.observation_width,
            height=self.observation_height,
        )
        env.set_task(mt1.train_tasks[0])
        if self.camera_name == "corner2":
            # Match LeRobot's MetaWorld dataset camera convention for corner2.
            env.model.cam_pos[2] = LEROBOT_CORNER2_CAMERA_POSITION
        env.max_path_length = self._max_episode_steps
        env.reset()
        env._freeze_rand_vec = False
        return env

    def render(self) -> np.ndarray:
        image = self._ensure_env().render()
        return self._normalize_image(image)

    def reset(
        self,
        seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = kwargs
        super().reset(seed=seed)
        backend = self._ensure_env()
        if seed is not None and hasattr(backend, "seed"):
            backend.seed(int(seed))
            if hasattr(backend, "seeded_rand_vec"):
                backend.seeded_rand_vec = True
        self._step_count = 0
        raw_obs, _info = backend.reset(seed=seed)
        return self._format_raw_obs(raw_obs), {"is_success": False}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=self.action_space.dtype)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, "
                f"got shape {action.shape}."
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("MetaWorld actions must be finite.")
        if not self.action_space.contains(action):
            action_space = cast(spaces.Box, self.action_space)
            raise ValueError(
                "MetaWorld actions must be within the normalized action space "
                f"[{action_space.low.min()}, {action_space.high.max()}]."
            )

        raw_obs, reward, done, truncated, info = self._ensure_env().step(action)
        is_success = bool(info.get("success", 0))
        terminated = bool(done) or is_success
        self._step_count += 1
        truncated = bool(truncated) or self._step_count >= self._max_episode_steps
        info.update(
            {
                "task": self.task,
                "done": bool(done),
                "is_success": is_success,
                "truncated": truncated,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        if terminated or truncated:
            info["final_info"] = {
                "task": self.task,
                "done": bool(done),
                "is_success": is_success,
                "truncated": truncated,
            }
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _format_raw_obs(self, raw_obs: np.ndarray) -> dict[str, Any]:
        if self.obs_type not in {"pixels", "pixels_agent_pos"}:
            raise ValueError(f"Unsupported MetaWorld obs_type: {self.obs_type!r}.")

        image = self.render().copy()
        if self.obs_type == "pixels":
            return {"pixels": image}

        raw_obs_array = np.asarray(raw_obs)
        return {
            "pixels": image,
            "agent_pos": raw_obs_array[:OBS_DIM],
        }

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        if self.camera_name == "corner2":
            image = np.flip(image, (0, 1))
        return np.asarray(image, dtype=np.uint8)
