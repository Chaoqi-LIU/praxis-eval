# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboMimic gymnasium wrapper around robosuite tasks."""

from __future__ import annotations

import os
from contextlib import suppress
from copy import deepcopy
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from praxis_eval.envs.robomimic.tasks import get_task_instruction

_DEFAULT_CAMERA_NAMES: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
_DEFAULT_STATE_PORTS: tuple[str, ...] = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def _configure_default_egl_device_for_uuid_cuda_visibility() -> None:
    """Avoid setting an EGL device id robosuite cannot validate."""
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") is not None:
        return

    mujoco_gl = os.environ.get("MUJOCO_GL")
    if mujoco_gl is not None and mujoco_gl.lower().strip() in {
        "disable",
        "disabled",
        "false",
        "glx",
        "off",
        "osmesa",
        "0",
    }:
        return

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return

    visible_device_tokens = [token.strip() for token in visible_devices.split(",")]
    if all(token.isdigit() for token in visible_device_tokens if token):
        os.environ["MUJOCO_EGL_DEVICE_ID"] = visible_device_tokens[0]


def _raw_observations(env: Any) -> dict[str, Any]:
    get_obs = env._get_observations
    try:
        return get_obs(force_update=True)
    except TypeError:
        return get_obs()


def _robomimic_controller_config(robot: str) -> dict[str, Any]:
    """Return the controller mode used by the official RoboMimic v1.5 PH demos."""
    from robosuite.controllers import load_composite_controller_config

    raw_config = deepcopy(
        load_composite_controller_config(
            controller=None,
            robot=robot,
        )
    )
    if raw_config is None:
        raise RuntimeError(f"Could not load robosuite controller config for {robot}.")
    controller_config = cast(dict[str, Any], raw_config)
    body_parts = controller_config.get("body_parts")
    if not isinstance(body_parts, dict) or not isinstance(
        body_parts.get("right"), dict
    ):
        raise RuntimeError(
            f"Robosuite controller config for {robot} has no right arm controller."
        )
    right_controller = cast(dict[str, Any], body_parts["right"])
    right_controller["input_type"] = "delta"
    right_controller["input_ref_frame"] = "world"
    return controller_config


def create_env(
    *,
    task_name: str,
    camera_names: list[str],
    image_size: int,
    enable_render: bool,
    seed: int,
    robot: str,
) -> Any:
    """Create the underlying robosuite environment lazily."""
    if enable_render:
        _configure_default_egl_device_for_uuid_cuda_visibility()

    import robosuite

    controller_config = _robomimic_controller_config(robot)
    env = robosuite.make(
        env_name=task_name,
        robots=robot,
        controller_configs=controller_config,
        control_freq=20,
        camera_names=camera_names,
        camera_widths=int(image_size),
        camera_heights=int(image_size),
        has_renderer=False,
        has_offscreen_renderer=bool(enable_render),
        ignore_done=True,
        reward_shaping=False,
        use_object_obs=True,
        use_camera_obs=bool(enable_render),
        camera_depths=False,
        lite_physics=False,
        seed=int(seed),
    )
    env.hard_reset = False
    return env


class RobomimicEnv(gym.Env):
    """Gymnasium wrapper around a RoboMimic robosuite task.

    Observation format mirrors the evaluator RoboCasa wrapper:
      - ``pixels``: ``{camera_name: HxWx3 uint8}``
      - ``robot_state``: dict keyed by robosuite state ports

    ``task_description`` is the natural-language policy instruction for the
    canonical RoboMimic task.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 20}
    _FLIP_AXIS = 0

    def __init__(
        self,
        task_name: str,
        image_size: int = 128,
        seed: int = 42,
        camera_names: list[str] | None = None,
        state_ports: list[str] | None = None,
        video_camera: str = "agentview",
        video_resolution: int = 512,
        max_episode_steps: int = 800,
        enable_render: bool = True,
        robot: str = "Panda",
    ) -> None:
        super().__init__()
        self.task_name = str(task_name)
        self.image_size = int(image_size)
        self._seed = int(seed)
        self.camera_names = list(camera_names or _DEFAULT_CAMERA_NAMES)
        self.state_ports = list(state_ports or _DEFAULT_STATE_PORTS)
        self.video_camera = str(video_camera)
        self.video_resolution = int(video_resolution)
        self._max_episode_steps = int(max_episode_steps)
        self.enable_render = bool(enable_render)
        self.robot = str(robot)
        self._done = False
        self._step_count = 0

        env_camera_names = list(dict.fromkeys([*self.camera_names, self.video_camera]))
        self.env = create_env(
            task_name=self.task_name,
            camera_names=env_camera_names,
            image_size=self.image_size,
            enable_render=self.enable_render,
            seed=self._seed,
            robot=self.robot,
        )
        raw_obs = _raw_observations(self.env)
        self.observation_space = self._build_observation_space(raw_obs)
        action_low, action_high = self.env.action_spec
        self.action_space = spaces.Box(
            low=np.asarray(action_low, dtype=np.float32),
            high=np.asarray(action_high, dtype=np.float32),
            shape=np.asarray(action_low).shape,
            dtype=np.float32,
        )

    @property
    def task_description(self) -> str:
        return get_task_instruction(self.task_name)

    @property
    def task(self) -> str:
        return self.task_name

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reset_options = {} if options is None else dict(options)
        current_seed = int(
            reset_options.pop(
                "episode_seed",
                self._seed if seed is None else int(seed),
            )
        )
        reseed_inner_env = bool(reset_options.pop("reseed_inner_env", True))
        if reseed_inner_env:
            self._prepare_inner_env_for_reset(seed=current_seed)
        raw_obs = self.env.reset()
        self._seed = current_seed
        self._done = False
        self._step_count = 0
        return self._extract_obs(raw_obs), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=self.action_space.dtype)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected RoboMimic action shape {self.action_space.shape}, "
                f"got {action.shape}."
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("RoboMimic actions must be finite.")
        if not self.action_space.contains(action):
            raise ValueError(
                "RoboMimic actions must be within the action space bounds."
            )

        self._step_count += 1
        raw_obs, _reward, raw_terminated, info = self.env.step(action)
        is_success = bool(self.env._check_success())
        reward = 1.0 if is_success else 0.0
        info = dict(info)
        info["is_success"] = is_success
        terminated = self._done or bool(raw_terminated) or is_success
        truncated = not terminated and self._step_count >= self._max_episode_steps
        self._done = terminated or truncated
        info["truncated"] = truncated
        if terminated or truncated:
            info["final_info"] = {
                "task": self.task_name,
                "done": bool(raw_terminated),
                "is_success": is_success,
                "truncated": truncated,
            }
        return self._extract_obs(raw_obs), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        frame = self.env.sim.render(
            height=self.video_resolution,
            width=self.video_resolution,
            camera_name=self.video_camera,
        )
        return np.flip(frame, axis=self._FLIP_AXIS).astype(np.uint8)

    def close(self) -> None:
        self.env.close()

    def _prepare_inner_env_for_reset(self, *, seed: int) -> None:
        seed_fn = getattr(self.env, "seed", None)
        if callable(seed_fn):
            seed_fn(int(seed))
        else:
            with suppress(Exception):
                self.env.seed = int(seed)
        if hasattr(self.env, "rng"):
            with suppress(Exception):
                self.env.rng = np.random.default_rng(int(seed))

    def _build_observation_space(self, raw_obs: dict[str, Any]) -> spaces.Dict:
        pixel_spaces: dict[str, gym.Space[Any]] = {
            cam: spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size, 3),
                dtype=np.uint8,
            )
            for cam in self.camera_names
        }
        state_spaces: dict[str, gym.Space[Any]] = {}
        for port in self.state_ports:
            if port not in raw_obs:
                raise KeyError(
                    f"RoboMimic raw observation is missing state port {port!r}. "
                    f"Available keys: {sorted(raw_obs)}"
                )
            state_spaces[port] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=np.asarray(raw_obs[port]).shape,
                dtype=np.float32,
            )
        return spaces.Dict(
            {
                "pixels": spaces.Dict(pixel_spaces),
                "robot_state": spaces.Dict(state_spaces),
            }
        )

    def _extract_obs(self, raw_obs: dict[str, Any] | None = None) -> dict[str, Any]:
        if raw_obs is None:
            raw_obs = _raw_observations(self.env)

        if self.enable_render:
            pixels = {
                cam: np.flip(raw_obs[f"{cam}_image"], axis=self._FLIP_AXIS).astype(
                    np.uint8
                )
                for cam in self.camera_names
            }
        else:
            pixels = {
                cam: np.zeros(
                    (self.image_size, self.image_size, 3),
                    dtype=np.uint8,
                )
                for cam in self.camera_names
            }
        robot_state = {
            port: np.asarray(raw_obs[port], dtype=np.float32)
            for port in self.state_ports
        }
        return {"pixels": pixels, "robot_state": robot_state}
