# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO registrations for the generic env factory."""

from __future__ import annotations

import importlib
from functools import partial
from typing import Any

from praxis_eval.envs.libero.spec import parse_camera_names
from praxis_eval.envs.libero.tasks import (
    infer_libero_eval_target_from_dataset,
    list_libero_tasks,
)


def build_libero_env(cfg_obj: Any, n_envs: int, use_async_envs: bool) -> Any:
    """Create LIBERO envs through the evaluator LeRobot adapter."""
    if use_async_envs:
        return _build_libero_env(
            cfg_obj,
            n_envs=n_envs,
            env_cls=_make_async_vector_env_cls(cfg_obj),
        )

    from gymnasium.vector import SyncVectorEnv

    return _build_libero_env(cfg_obj, n_envs=n_envs, env_cls=SyncVectorEnv)


def build_libero_async_env(cfg_obj: Any, n_envs: int) -> Any:
    """Create LIBERO async envs with a dummy-env bootstrap for space inference."""
    return _build_libero_env(
        cfg_obj,
        n_envs=n_envs,
        env_cls=_make_async_vector_env_cls(cfg_obj),
    )


def _make_async_vector_env_cls(cfg_obj: Any) -> Any:
    from praxis_eval.envs.async_vector_env import AsyncVectorEnv
    from praxis_eval.envs.libero import make_dummy_libero_env_fn

    camera_names = parse_camera_names(cfg_obj.camera_name)
    # Keep dummy-space bootstrap aligned with explicit LiberoEnv config fields.
    dummy_observation_height = int(cfg_obj.observation_height)
    dummy_observation_width = int(cfg_obj.observation_width)
    dummy_env_fn = make_dummy_libero_env_fn(
        camera_names=camera_names,
        obs_type=cfg_obj.obs_type,
        observation_height=dummy_observation_height,
        observation_width=dummy_observation_width,
        camera_name_mapping=cfg_obj.camera_name_mapping,
    )
    return partial(AsyncVectorEnv, dummy_env_fn=dummy_env_fn)


def _build_libero_env(cfg_obj: Any, *, n_envs: int, env_cls: Any) -> Any:
    # Keep real env constructor kwargs aligned with dummy_env_fn so space checks
    # compare like-for-like (especially observation width/height and key mapping).
    env_gym_kwargs = dict(cfg_obj.gym_kwargs or {})
    env_gym_kwargs.setdefault("num_steps_wait", 20)
    env_gym_kwargs.setdefault("obs_type", cfg_obj.obs_type)
    env_gym_kwargs.setdefault("render_mode", cfg_obj.render_mode)
    env_gym_kwargs["observation_height"] = int(cfg_obj.observation_height)
    env_gym_kwargs["observation_width"] = int(cfg_obj.observation_width)
    env_gym_kwargs["camera_name_mapping"] = cfg_obj.camera_name_mapping

    env_module = importlib.import_module("praxis_eval.envs.libero.env")
    return env_module.create_libero_envs(
        task=cfg_obj.task,
        n_envs=n_envs,
        camera_name=cfg_obj.camera_name,
        init_states=cfg_obj.init_states,
        gym_kwargs=env_gym_kwargs,
        env_cls=env_cls,
        control_mode=cfg_obj.control_mode,
        episode_length=cfg_obj.episode_length,
    )


def register_libero_env_family() -> None:
    """Register LIBERO config, task, and eval-pool hooks."""
    from praxis_eval.envs.factory import (
        register_async_env_builder,
        register_env_builder,
        register_env_config,
        register_eval_pool_builder,
        register_eval_target_inferer,
        register_task_lister,
    )

    register_env_config("libero", "praxis_eval.envs.libero.config:LiberoEnvConfig")
    register_env_builder("libero", build_libero_env)
    register_async_env_builder("libero", build_libero_async_env)
    register_task_lister("libero", list_libero_tasks)
    register_eval_target_inferer("libero", infer_libero_eval_target_from_dataset)
    register_eval_pool_builder(
        "libero",
        "praxis_eval.envs.libero.eval:build_libero_eval_pool",
    )
