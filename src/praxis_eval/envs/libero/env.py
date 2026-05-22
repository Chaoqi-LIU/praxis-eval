# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO adapter around LeRobot's environment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import gymnasium as gym

from praxis_eval.envs.libero.config import ensure_libero_config
from praxis_eval.envs.libero.spec import (
    DEFAULT_CAMERA_NAME,
    NOOP_ACTION,
    parse_camera_names,
    select_task_ids,
)

ensure_libero_config()
from lerobot.envs.libero import LiberoEnv as _LeRobotLiberoEnv  # noqa: E402


def get_suite(suite_name: str) -> Any:
    """Instantiate a LIBERO benchmark suite by name."""
    ensure_libero_config()
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        available = ", ".join(sorted(benchmark_dict.keys()))
        raise ValueError(f"Unknown LIBERO suite {suite_name!r}. Available: {available}")

    suite = benchmark_dict[suite_name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"LIBERO suite {suite_name!r} has no tasks.")
    return suite


def _right_arm_controller(robot: Any) -> Any:
    part_controllers = getattr(robot, "part_controllers", None)
    if isinstance(part_controllers, Mapping) and "right" in part_controllers:
        return part_controllers["right"]

    controller = getattr(robot, "controller", None)
    if controller is None:
        raise AttributeError(
            "LIBERO robot exposes neither part_controllers['right'] nor controller."
        )
    return controller


def _controller_orientation_matrix(controller: Any) -> Any:
    if hasattr(controller, "ee_ori_mat"):
        return controller.ee_ori_mat
    if hasattr(controller, "ref_ori_mat"):
        return controller.ref_ori_mat
    raise AttributeError(
        "LIBERO controller exposes neither ee_ori_mat nor ref_ori_mat."
    )


class LiberoEnv(_LeRobotLiberoEnv):
    """LeRobot LIBERO env adapted to the Praxis robosuite fork."""

    def reset(
        self,
        seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = kwargs
        gym.Env.reset(self, seed=seed)
        self._env.seed(seed)
        raw_obs = self._env.reset()
        if self.init_states and self._init_states is not None:
            raw_obs = self._env.set_init_state(
                self._init_states[self.init_state_id % len(self._init_states)]
            )
            self.init_state_id += self._reset_stride

        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(NOOP_ACTION)

        if self.control_mode == "absolute":
            use_delta = False
        elif self.control_mode == "relative":
            use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")

        for robot in self._env.robots:
            _right_arm_controller(robot).use_delta = use_delta

        return self._format_raw_obs(raw_obs), {"is_success": False}

    def _format_raw_obs(self, raw_obs: Mapping[str, Any]) -> dict[str, Any]:
        images = {
            self.camera_name_mapping[camera_name]: raw_obs[camera_name]
            for camera_name in self.camera_name
        }
        if self.obs_type == "pixels":
            return {"pixels": images.copy()}

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        if eef_pos is None or eef_quat is None or gripper_qpos is None:
            raise ValueError(
                "Missing required LIBERO robot state fields: "
                f"eef_pos={eef_pos is not None}, "
                f"eef_quat={eef_quat is not None}, "
                f"gripper_qpos={gripper_qpos is not None}."
            )

        controller = _right_arm_controller(self._env.robots[0])
        return {
            "pixels": images,
            "robot_state": {
                "eef": {
                    "pos": eef_pos,
                    "quat": eef_quat,
                    "mat": _controller_orientation_matrix(controller),
                },
                "gripper": {
                    "qpos": gripper_qpos,
                    "qvel": raw_obs.get("robot0_gripper_qvel"),
                },
                "joints": {
                    "pos": raw_obs.get("robot0_joint_pos"),
                    "vel": raw_obs.get("robot0_joint_vel"),
                },
            },
        }


def _make_env_fns(
    *,
    suite: Any,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    control_mode: str,
) -> list[Callable[[], LiberoEnv]]:
    def _make_env(episode_index: int, **kwargs: Any) -> LiberoEnv:
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            n_envs=n_envs,
            control_mode=control_mode,
            **dict(kwargs),
        )

    return [
        partial(_make_env, episode_index, **dict(gym_kwargs))
        for episode_index in range(n_envs)
    ]


def create_libero_envs(
    *,
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = DEFAULT_CAMERA_NAME,
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    control_mode: str = "relative",
    episode_length: int | None = None,
) -> dict[str, dict[int, Any]]:
    """Create vectorized LIBERO envs as ``{suite_name: {task_id: vec_env}}``."""
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable env-vector constructor.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int, got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)
    camera_names = parse_camera_names(camera_name)
    suite_names = [suite.strip() for suite in str(task).split(",") if suite.strip()]
    if not suite_names:
        raise ValueError("task must contain at least one LIBERO suite name.")

    envs: dict[str, dict[int, Any]] = defaultdict(dict)
    for suite_name in suite_names:
        suite = get_suite(suite_name)
        selected_ids = select_task_ids(len(suite.tasks), task_ids_filter)
        if not selected_ids:
            raise ValueError(f"No LIBERO tasks selected for suite {suite_name!r}.")

        for task_id in selected_ids:
            envs[suite_name][int(task_id)] = env_cls(
                _make_env_fns(
                    suite=suite,
                    suite_name=suite_name,
                    task_id=int(task_id),
                    n_envs=n_envs,
                    camera_names=camera_names,
                    episode_length=episode_length,
                    init_states=init_states,
                    gym_kwargs=gym_kwargs,
                    control_mode=control_mode,
                )
            )

    return {suite_name: dict(task_envs) for suite_name, task_envs in envs.items()}
