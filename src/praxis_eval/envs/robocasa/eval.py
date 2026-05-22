# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa persistent eval pool helpers."""

from __future__ import annotations

import logging
import multiprocessing
from typing import Any, cast

from praxis_eval.envs.async_vector_env import AsyncVectorEnv
from praxis_eval.envs.eval_pool import EvalLaneJob, EvalPoolHandle
from praxis_eval.envs.robocasa.tasks import get_subtasks, get_task_horizon

_ROBOCASA_MP_CONTEXT = "fork"
_ROBOCASA_BUILD_TIMEOUT_SEC = 600.0


def _resolve_robocasa_mp_context() -> str:
    """Return the multiprocessing context for RoboCasa eval workers."""
    available = set(multiprocessing.get_all_start_methods())
    if _ROBOCASA_MP_CONTEXT in available:
        return _ROBOCASA_MP_CONTEXT
    if "spawn" in available:
        logging.getLogger(__name__).warning(
            "RoboCasa preferred multiprocessing context %r is unavailable on this platform (%s); using %r.",
            _ROBOCASA_MP_CONTEXT,
            ", ".join(sorted(available)),
            "spawn",
        )
        return "spawn"

    fallback = multiprocessing.get_start_method(allow_none=False)
    logging.getLogger(__name__).warning(
        "No preferred RoboCasa multiprocessing context available; falling back to %r.",
        fallback,
    )
    return fallback


def build_robocasa_eval_pool(
    cfg_obj: Any,
    tasks: list[tuple[str, int]],
    n_envs: int,
    debug_verbose: bool = False,
) -> EvalPoolHandle:
    """Create one persistent async eval pool for RoboCasa.

    Args:
        cfg_obj: A ``RobocasaEnvConfig`` instance.
        tasks: List of ``(task_name, task_id)`` pairs from ``list_robocasa_tasks``.
        n_envs: Number of parallel worker envs.
        debug_verbose: If True, suppress less robocasa startup noise.
    """
    from praxis_eval.envs.robocasa.runtime import make_dummy_robocasa_env_fn

    if not tasks:
        raise ValueError("tasks must be non-empty for build_robocasa_eval_pool().")

    camera_names: list[str] = list(
        getattr(
            cfg_obj,
            "camera_names",
            [
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ],
        )
    )
    image_size: int = int(getattr(cfg_obj, "image_size", 128))
    max_episode_steps: int = int(getattr(cfg_obj, "max_episode_steps", 500))
    split: str = str(getattr(cfg_obj, "split", "all"))

    dummy_env_fn = make_dummy_robocasa_env_fn(
        camera_names=camera_names,
        image_size=image_size,
    )
    mp_context = _resolve_robocasa_mp_context()
    handle = EvalPoolHandle(
        env_pool=None,
        num_envs=n_envs,
        prepare_jobs=lambda lane_jobs: _prepare_robocasa_eval_jobs(
            handle,
            lane_jobs,
            split=split,
            camera_names=camera_names,
            image_size=image_size,
            max_episode_steps=max_episode_steps,
            dummy_env_fn=dummy_env_fn,
            mp_context=mp_context,
            build_timeout_sec=_ROBOCASA_BUILD_TIMEOUT_SEC,
        ),
    )
    return handle


def _make_lane_env_fn(
    *,
    lane_idx: int,
    lane_job: EvalLaneJob,
    split: str,
    camera_names: list[str],
    image_size: int,
    max_episode_steps: int,
):
    from praxis_eval.envs.robocasa.runtime import (
        build_robocasa_env_with_retries,
        construct_robocasa_eval_lane,
    )

    lane_task_group = str(lane_job.task_group)
    subtasks = get_subtasks(lane_task_group)
    task_name = subtasks[int(lane_job.task_id) % len(subtasks)]
    seed = int(lane_job.episode_index)
    task_horizon = get_task_horizon(task_name, default=max_episode_steps)

    return lambda: construct_robocasa_eval_lane(
        lambda: build_robocasa_env_with_retries(
            task_name=task_name,
            split=split,
            image_size=image_size,
            seed=seed,
            camera_names=camera_names,
            max_episode_steps=task_horizon,
            enable_render=True,
        ),
        task_group=lane_task_group,
        lane_idx=lane_idx,
    )


def _build_env_pool_for_lane_jobs(
    lane_jobs: list[EvalLaneJob],
    *,
    split: str,
    camera_names: list[str],
    image_size: int,
    max_episode_steps: int,
    dummy_env_fn: Any,
    mp_context: str,
) -> AsyncVectorEnv:
    env_fns = [
        _make_lane_env_fn(
            lane_idx=lane_idx,
            lane_job=lane_job,
            split=split,
            camera_names=camera_names,
            image_size=image_size,
            max_episode_steps=max_episode_steps,
        )
        for lane_idx, lane_job in enumerate(lane_jobs)
    ]
    return AsyncVectorEnv(
        env_fns,
        dummy_env_fn=dummy_env_fn,
        context=mp_context,
    )


def _prepare_robocasa_eval_jobs(
    handle: EvalPoolHandle,
    lane_jobs: list[EvalLaneJob | None],
    *,
    split: str,
    camera_names: list[str],
    image_size: int,
    max_episode_steps: int,
    dummy_env_fn: Any,
    mp_context: str,
    build_timeout_sec: float,
) -> None:
    first_job = next((job for job in lane_jobs if job is not None), None)
    if first_job is None:
        return

    prepared: list[EvalLaneJob] = [
        first_job if job is None else job for job in lane_jobs
    ]
    env_pool = handle.env_pool
    if env_pool is None:
        handle.env_pool = _build_env_pool_for_lane_jobs(
            prepared,
            split=split,
            camera_names=camera_names,
            image_size=image_size,
            max_episode_steps=max_episode_steps,
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
