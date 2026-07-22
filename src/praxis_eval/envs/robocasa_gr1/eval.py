# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa GR-1 persistent async evaluation pool."""

from __future__ import annotations

import logging
import multiprocessing
from typing import Any, cast

from praxis_eval.envs.async_vector_env import AsyncVectorEnv
from praxis_eval.envs.eval_pool import EvalLaneJob, EvalPoolHandle
from praxis_eval.envs.robocasa_gr1.env import RobocasaGr1Env
from praxis_eval.envs.robocasa_gr1.runtime import (
    construct_robocasa_gr1_eval_lane,
    make_dummy_robocasa_gr1_env_fn,
)

_GR1_MP_CONTEXT = "fork"
_GR1_BUILD_TIMEOUT_SEC = 600.0


def _resolve_gr1_mp_context() -> str:
    available = set(multiprocessing.get_all_start_methods())
    if _GR1_MP_CONTEXT in available:
        return _GR1_MP_CONTEXT
    if "spawn" in available:
        logging.getLogger(__name__).warning(
            "RoboCasa GR-1 multiprocessing context 'fork' is unavailable; using spawn."
        )
        return "spawn"
    return multiprocessing.get_start_method(allow_none=False)


def build_robocasa_gr1_eval_pool(
    cfg_obj: Any,
    tasks: list[tuple[str, int]],
    n_envs: int,
    debug_verbose: bool = False,
) -> EvalPoolHandle:
    """Create a lazy worker pool; the parent process never imports GR-1."""
    _ = debug_verbose
    if not tasks:
        raise ValueError("tasks must be non-empty for RoboCasa GR-1 evaluation.")
    max_steps = int(getattr(cfg_obj, "max_episode_steps", 720))
    dummy_env_fn = make_dummy_robocasa_gr1_env_fn()
    mp_context = _resolve_gr1_mp_context()
    handle = EvalPoolHandle(
        env_pool=None,
        num_envs=n_envs,
        prepare_jobs=lambda lane_jobs: _prepare_jobs(
            handle,
            lane_jobs,
            max_steps=max_steps,
            dummy_env_fn=dummy_env_fn,
            mp_context=mp_context,
        ),
    )
    return handle


def _make_lane_env_fn(
    *, task_name: str, lane_idx: int, max_steps: int, initial_seed: int
) -> Any:
    return lambda: construct_robocasa_gr1_eval_lane(
        lambda: RobocasaGr1Env(
            task_name,
            max_episode_steps=max_steps,
            enable_render=True,
        ),
        lane_idx=lane_idx,
        initial_seed=initial_seed,
    )


def _build_pool(
    jobs: list[EvalLaneJob],
    *,
    max_steps: int,
    dummy_env_fn: Any,
    mp_context: str,
) -> AsyncVectorEnv:
    return AsyncVectorEnv(
        [
            _make_lane_env_fn(
                task_name=str(job.task_group),
                lane_idx=lane_idx,
                max_steps=max_steps,
                initial_seed=int(job.episode_index),
            )
            for lane_idx, job in enumerate(jobs)
        ],
        dummy_env_fn=dummy_env_fn,
        context=mp_context,
    )


def _prepare_jobs(
    handle: EvalPoolHandle,
    lane_jobs: list[EvalLaneJob | None],
    *,
    max_steps: int,
    dummy_env_fn: Any,
    mp_context: str,
) -> None:
    first_job = next((job for job in lane_jobs if job is not None), None)
    if first_job is None:
        return
    jobs = [first_job if job is None else job for job in lane_jobs]
    if handle.env_pool is None:
        handle.env_pool = _build_pool(
            jobs,
            max_steps=max_steps,
            dummy_env_fn=dummy_env_fn,
            mp_context=mp_context,
        )
        return

    pool = cast(AsyncVectorEnv, handle.env_pool)
    args_list = [(job.task_id, job.episode_index, job.task_group) for job in jobs]
    try:
        pool.call_each(
            "prepare_eval_job",
            args_list=args_list,
            timeout=_GR1_BUILD_TIMEOUT_SEC,
        )
    except Exception:
        try:
            pool.close(terminate=True)
        finally:
            handle.env_pool = None
        raise
