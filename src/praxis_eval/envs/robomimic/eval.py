# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboMimic persistent eval pool helpers."""

from __future__ import annotations

import logging
import multiprocessing
from typing import Any, cast

from praxis_eval.envs.async_vector_env import AsyncVectorEnv
from praxis_eval.envs.eval_pool import EvalLaneJob, EvalPoolHandle
from praxis_eval.envs.robomimic.tasks import get_subtasks, get_task_horizon

_ROBOMIMIC_MP_CONTEXT = "fork"
_ROBOMIMIC_BUILD_TIMEOUT_SEC = 600.0


def _resolve_robomimic_mp_context() -> str:
    """Return the multiprocessing context for RoboMimic eval workers."""
    available = set(multiprocessing.get_all_start_methods())
    if _ROBOMIMIC_MP_CONTEXT in available:
        return _ROBOMIMIC_MP_CONTEXT
    if "spawn" in available:
        logging.getLogger(__name__).warning(
            "RoboMimic preferred multiprocessing context %r is unavailable on this "
            "platform (%s); using %r.",
            _ROBOMIMIC_MP_CONTEXT,
            ", ".join(sorted(available)),
            "spawn",
        )
        return "spawn"
    fallback = multiprocessing.get_start_method(allow_none=False)
    logging.getLogger(__name__).warning(
        "No preferred RoboMimic multiprocessing context available; falling back to %r.",
        fallback,
    )
    return fallback


def build_robomimic_eval_pool(
    cfg_obj: Any,
    tasks: list[tuple[str, int]],
    n_envs: int,
    debug_verbose: bool = False,
) -> EvalPoolHandle:
    """Create one persistent async eval pool for RoboMimic."""
    from praxis_eval.envs.robomimic.runtime import make_dummy_robomimic_env_fn

    _ = debug_verbose
    if not tasks:
        raise ValueError("tasks must be non-empty for build_robomimic_eval_pool().")

    camera_names = list(getattr(cfg_obj, "camera_names", ["agentview"]))
    state_ports = list(
        getattr(
            cfg_obj,
            "state_ports",
            ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        )
    )
    image_size = int(getattr(cfg_obj, "image_size", 128))
    video_camera = str(getattr(cfg_obj, "video_camera", camera_names[0]))
    video_resolution = int(getattr(cfg_obj, "video_resolution", 512))
    max_episode_steps = int(getattr(cfg_obj, "max_episode_steps", 800))
    robot = str(getattr(cfg_obj, "robot", "Panda"))

    dummy_env_fn = make_dummy_robomimic_env_fn(
        camera_names=camera_names,
        image_size=image_size,
        state_ports=state_ports,
    )
    mp_context = _resolve_robomimic_mp_context()
    handle = EvalPoolHandle(
        env_pool=None,
        num_envs=n_envs,
        prepare_jobs=lambda lane_jobs: _prepare_robomimic_eval_jobs(
            handle,
            lane_jobs,
            camera_names=camera_names,
            state_ports=state_ports,
            image_size=image_size,
            video_camera=video_camera,
            video_resolution=video_resolution,
            max_episode_steps=max_episode_steps,
            robot=robot,
            dummy_env_fn=dummy_env_fn,
            mp_context=mp_context,
            build_timeout_sec=_ROBOMIMIC_BUILD_TIMEOUT_SEC,
        ),
    )
    return handle


def _make_lane_env_fn(
    *,
    lane_idx: int,
    lane_job: EvalLaneJob,
    camera_names: list[str],
    state_ports: list[str],
    image_size: int,
    video_camera: str,
    video_resolution: int,
    max_episode_steps: int,
    robot: str,
):
    from praxis_eval.envs.robomimic.runtime import (
        build_robomimic_env_with_retries,
        construct_robomimic_eval_lane,
    )

    lane_task_group = str(lane_job.task_group)
    subtasks = get_subtasks(lane_task_group)
    task_name = subtasks[int(lane_job.task_id) % len(subtasks)]
    seed = int(lane_job.episode_index)
    task_horizon = get_task_horizon(task_name, default=max_episode_steps)

    return lambda: construct_robomimic_eval_lane(
        lambda: build_robomimic_env_with_retries(
            task_name=task_name,
            image_size=image_size,
            seed=seed,
            camera_names=camera_names,
            state_ports=state_ports,
            video_camera=video_camera,
            video_resolution=video_resolution,
            max_episode_steps=task_horizon,
            enable_render=True,
            robot=robot,
        ),
        task_group=lane_task_group,
        lane_idx=lane_idx,
    )


def _build_env_pool_for_lane_jobs(
    lane_jobs: list[EvalLaneJob],
    *,
    camera_names: list[str],
    state_ports: list[str],
    image_size: int,
    video_camera: str,
    video_resolution: int,
    max_episode_steps: int,
    robot: str,
    dummy_env_fn: Any,
    mp_context: str,
) -> AsyncVectorEnv:
    env_fns = [
        _make_lane_env_fn(
            lane_idx=lane_idx,
            lane_job=lane_job,
            camera_names=camera_names,
            state_ports=state_ports,
            image_size=image_size,
            video_camera=video_camera,
            video_resolution=video_resolution,
            max_episode_steps=max_episode_steps,
            robot=robot,
        )
        for lane_idx, lane_job in enumerate(lane_jobs)
    ]
    return AsyncVectorEnv(
        env_fns,
        dummy_env_fn=dummy_env_fn,
        context=mp_context,
    )


def _prepare_robomimic_eval_jobs(
    handle: EvalPoolHandle,
    lane_jobs: list[EvalLaneJob | None],
    *,
    camera_names: list[str],
    state_ports: list[str],
    image_size: int,
    video_camera: str,
    video_resolution: int,
    max_episode_steps: int,
    robot: str,
    dummy_env_fn: Any,
    mp_context: str,
    build_timeout_sec: float,
) -> None:
    first_job = next((job for job in lane_jobs if job is not None), None)
    if first_job is None:
        return

    prepared = [first_job if job is None else job for job in lane_jobs]
    env_pool = handle.env_pool
    if env_pool is None:
        handle.env_pool = _build_env_pool_for_lane_jobs(
            prepared,
            camera_names=camera_names,
            state_ports=state_ports,
            image_size=image_size,
            video_camera=video_camera,
            video_resolution=video_resolution,
            max_episode_steps=max_episode_steps,
            robot=robot,
            dummy_env_fn=dummy_env_fn,
            mp_context=mp_context,
        )
        return

    async_env_pool = cast(AsyncVectorEnv, env_pool)
    args_list = [(job.task_id, job.episode_index, job.task_group) for job in prepared]
    try:
        async_env_pool.call_each(
            "prepare_eval_job",
            args_list=args_list,
            timeout=build_timeout_sec,
        )
    except Exception:
        try:
            async_env_pool.close(terminate=True)
        finally:
            handle.env_pool = None
        raise
