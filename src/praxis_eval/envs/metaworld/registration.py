# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""MetaWorld registrations for the generic env factory."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Any

from praxis_eval.envs.metaworld.tasks import (
    infer_metaworld_eval_target_from_dataset,
    list_metaworld_tasks,
    resolve_task_name,
)


def build_metaworld_env(cfg_obj: Any, n_envs: int, use_async_envs: bool) -> Any:
    """Create MetaWorld envs through the evaluator local env class."""
    if use_async_envs:
        env_cls = _make_async_vector_env_cls(cfg_obj)
    else:
        from gymnasium.vector import SyncVectorEnv

        env_cls = SyncVectorEnv

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for task_group, task_id in list_metaworld_tasks({}, cfg_obj):
        task_name = resolve_task_name(task_group, task_id)
        out[str(task_group)][int(task_id)] = env_cls(
            [
                _make_metaworld_env_fn(cfg_obj, task_name=task_name)
                for _ in range(n_envs)
            ]
        )
    return {task_group: dict(task_envs) for task_group, task_envs in out.items()}


def build_metaworld_async_env(cfg_obj: Any, n_envs: int) -> Any:
    """Create async MetaWorld envs through the evaluator local env class."""
    return build_metaworld_env(cfg_obj, n_envs=n_envs, use_async_envs=True)


def _make_async_vector_env_cls(cfg_obj: Any) -> Any:
    from praxis_eval.envs.async_vector_env import AsyncVectorEnv
    from praxis_eval.envs.metaworld.runtime import make_dummy_metaworld_env_fn

    dummy_env_fn = make_dummy_metaworld_env_fn(
        obs_type=str(getattr(cfg_obj, "obs_type", "pixels_agent_pos")),
        camera_name=str(getattr(cfg_obj, "camera_name", "corner2")),
        observation_height=int(getattr(cfg_obj, "observation_height", 480)),
        observation_width=int(getattr(cfg_obj, "observation_width", 480)),
    )
    return partial(AsyncVectorEnv, dummy_env_fn=dummy_env_fn)


def _make_metaworld_env_fn(cfg_obj: Any, *, task_name: str) -> Any:
    from praxis_eval.envs.metaworld.runtime import make_metaworld_env_fn

    return make_metaworld_env_fn(
        task_name=task_name,
        camera_name=str(getattr(cfg_obj, "camera_name", "corner2")),
        obs_type=str(getattr(cfg_obj, "obs_type", "pixels_agent_pos")),
        render_mode=str(getattr(cfg_obj, "render_mode", "rgb_array")),
        observation_width=int(getattr(cfg_obj, "observation_width", 480)),
        observation_height=int(getattr(cfg_obj, "observation_height", 480)),
        visualization_width=int(getattr(cfg_obj, "visualization_width", 640)),
        visualization_height=int(getattr(cfg_obj, "visualization_height", 480)),
        episode_length=getattr(cfg_obj, "episode_length", None),
    )


def register_metaworld_env_family() -> None:
    """Register MetaWorld config, task, and eval-pool hooks."""
    from praxis_eval.envs.factory import (
        register_async_env_builder,
        register_env_builder,
        register_env_config,
        register_eval_pool_builder,
        register_eval_target_inferer,
        register_task_lister,
    )

    register_env_config(
        "metaworld", "praxis_eval.envs.metaworld.config:MetaworldEnvConfig"
    )
    register_env_builder("metaworld", build_metaworld_env)
    register_async_env_builder("metaworld", build_metaworld_async_env)
    register_task_lister("metaworld", list_metaworld_tasks)
    register_eval_target_inferer("metaworld", infer_metaworld_eval_target_from_dataset)
    register_eval_pool_builder(
        "metaworld",
        "praxis_eval.envs.metaworld.eval:build_metaworld_eval_pool",
    )
