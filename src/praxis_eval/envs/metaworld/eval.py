# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""MetaWorld persistent eval pool helpers."""

from __future__ import annotations

from typing import Any, cast

from praxis_eval.envs.async_vector_env import AsyncVectorEnv
from praxis_eval.envs.eval_pool import EvalLaneJob, EvalPoolHandle
from praxis_eval.envs.metaworld.tasks import resolve_task_name


def build_metaworld_eval_pool(
    cfg_obj: Any,
    tasks: list[tuple[str, int]],
    n_envs: int,
    debug_verbose: bool = False,
) -> EvalPoolHandle:
    """Create one persistent async eval pool for MetaWorld."""
    from praxis_eval.envs.metaworld.runtime import make_dummy_metaworld_env_fn

    _ = (tasks, debug_verbose)
    dummy_env_fn = make_dummy_metaworld_env_fn(
        obs_type=str(getattr(cfg_obj, "obs_type", "pixels_agent_pos")),
        observation_height=int(getattr(cfg_obj, "observation_height", 480)),
        observation_width=int(getattr(cfg_obj, "observation_width", 480)),
    )
    handle = EvalPoolHandle(
        env_pool=None,
        num_envs=n_envs,
        prepare_jobs=lambda lane_jobs: _prepare_metaworld_eval_jobs(
            handle,
            lane_jobs,
            camera_name=str(getattr(cfg_obj, "camera_name", "corner2")),
            obs_type=str(getattr(cfg_obj, "obs_type", "pixels_agent_pos")),
            render_mode=str(getattr(cfg_obj, "render_mode", "rgb_array")),
            observation_width=int(getattr(cfg_obj, "observation_width", 480)),
            observation_height=int(getattr(cfg_obj, "observation_height", 480)),
            visualization_width=int(getattr(cfg_obj, "visualization_width", 640)),
            visualization_height=int(getattr(cfg_obj, "visualization_height", 480)),
            episode_length=getattr(cfg_obj, "episode_length", None),
            dummy_env_fn=dummy_env_fn,
        ),
    )
    return handle


def _make_lane_env_fn(
    *,
    lane_idx: int,
    lane_job: EvalLaneJob,
    camera_name: str,
    obs_type: str,
    render_mode: str,
    observation_width: int,
    observation_height: int,
    visualization_width: int,
    visualization_height: int,
    episode_length: int | None,
):
    from praxis_eval.envs.metaworld.runtime import (
        construct_metaworld_eval_lane,
        make_metaworld_env_fn,
    )

    task_name = resolve_task_name(lane_job.task_group, lane_job.task_id)
    base_env_fn = make_metaworld_env_fn(
        task_name=task_name,
        camera_name=camera_name,
        obs_type=obs_type,
        render_mode=render_mode,
        observation_width=observation_width,
        observation_height=observation_height,
        visualization_width=visualization_width,
        visualization_height=visualization_height,
        episode_length=episode_length,
    )
    return lambda: construct_metaworld_eval_lane(
        base_env_fn,
        task_group=str(lane_job.task_group),
        lane_idx=lane_idx,
    )


def _build_env_pool_for_lane_jobs(
    lane_jobs: list[EvalLaneJob],
    *,
    camera_name: str,
    obs_type: str,
    render_mode: str,
    observation_width: int,
    observation_height: int,
    visualization_width: int,
    visualization_height: int,
    episode_length: int | None,
    dummy_env_fn: Any,
) -> AsyncVectorEnv:
    env_fns = [
        _make_lane_env_fn(
            lane_idx=lane_idx,
            lane_job=lane_job,
            camera_name=camera_name,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
            episode_length=episode_length,
        )
        for lane_idx, lane_job in enumerate(lane_jobs)
    ]
    return AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)


def _prepare_metaworld_eval_jobs(
    handle: EvalPoolHandle,
    lane_jobs: list[EvalLaneJob | None],
    *,
    camera_name: str,
    obs_type: str,
    render_mode: str,
    observation_width: int,
    observation_height: int,
    visualization_width: int,
    visualization_height: int,
    episode_length: int | None,
    dummy_env_fn: Any,
) -> None:
    first_job = next((job for job in lane_jobs if job is not None), None)
    if first_job is None:
        return

    prepared = [first_job if job is None else job for job in lane_jobs]
    if handle.env_pool is None:
        handle.env_pool = _build_env_pool_for_lane_jobs(
            prepared,
            camera_name=camera_name,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
            episode_length=episode_length,
            dummy_env_fn=dummy_env_fn,
        )
        return

    env_pool = cast(AsyncVectorEnv, handle.env_pool)
    args_list = [(int(job.task_id), int(job.episode_index)) for job in prepared]
    kwargs_list = [{"task_group": str(job.task_group)} for job in prepared]
    env_pool.call_each("prepare_eval_job", args_list=args_list, kwargs_list=kwargs_list)
