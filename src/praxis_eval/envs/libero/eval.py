# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO-owned persistent eval pool helpers."""

from __future__ import annotations

import importlib
from typing import Any, cast

from praxis_eval.envs.async_vector_env import AsyncVectorEnv
from praxis_eval.envs.eval_pool import EvalLaneJob, EvalPoolHandle
from praxis_eval.envs.libero.output import suppress_libero_output
from praxis_eval.envs.libero.spec import parse_camera_names


def make_libero_eval_pool(
    env_cfg: Any,
    *,
    n_envs: int,
    debug_verbose: bool = False,
) -> EvalPoolHandle:
    """Create one persistent async vec pool for LIBERO eval lanes."""
    cfg_obj = _coerce_libero_env_config(env_cfg)
    if str(cfg_obj.type).strip().lower() != "libero":
        raise ValueError(
            "make_libero_eval_pool currently supports only env.type=libero."
        )
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}.")

    camera_names = parse_camera_names(cfg_obj.camera_name)

    env_gym_kwargs = dict(cfg_obj.gym_kwargs or {})
    env_gym_kwargs.setdefault("num_steps_wait", 20)
    env_gym_kwargs.pop("task_ids", None)
    env_gym_kwargs.setdefault("obs_type", cfg_obj.obs_type)
    env_gym_kwargs.setdefault("render_mode", cfg_obj.render_mode)
    env_gym_kwargs["observation_height"] = int(cfg_obj.observation_height)
    env_gym_kwargs["observation_width"] = int(cfg_obj.observation_width)
    env_gym_kwargs["camera_name_mapping"] = cfg_obj.camera_name_mapping

    from praxis_eval.envs.libero.runtime import make_dummy_libero_env_fn

    dummy_env_fn = make_dummy_libero_env_fn(
        camera_names=camera_names,
        obs_type=cfg_obj.obs_type,
        observation_height=int(cfg_obj.observation_height),
        observation_width=int(cfg_obj.observation_width),
        camera_name_mapping=cfg_obj.camera_name_mapping,
    )
    handle: EvalPoolHandle
    handle = EvalPoolHandle(
        env_pool=None,
        num_envs=n_envs,
        prepare_jobs=lambda lane_jobs: _prepare_libero_eval_jobs(
            handle,
            lane_jobs,
            camera_names=camera_names,
            episode_length=cfg_obj.episode_length,
            init_states=cfg_obj.init_states,
            gym_kwargs=env_gym_kwargs,
            control_mode=cfg_obj.control_mode,
            dummy_env_fn=dummy_env_fn,
            debug_verbose=debug_verbose,
        ),
    )
    return handle


def _coerce_libero_env_config(env_cfg: Any) -> Any:
    if str(getattr(env_cfg, "type", "")).strip().lower() == "libero":
        return env_cfg

    from praxis_eval.envs.factory import build_env_config

    return cast(Any, build_env_config(env_cfg))


def _make_lane_env_fn(
    *,
    lane_idx: int,
    lane_job: EvalLaneJob,
    suite: Any,
    suite_name: str,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: dict[str, Any],
    control_mode: str,
    reset_stride: int,
    debug_verbose: bool,
):
    from praxis_eval.envs.libero.runtime import (
        construct_libero_eval_lane,
        make_libero_env_fn,
    )

    base_env_fn = make_libero_env_fn(
        suite=suite,
        suite_name=suite_name,
        task_id=int(lane_job.task_id),
        episode_index=int(lane_job.episode_index),
        reset_stride=int(reset_stride),
        camera_names=camera_names,
        episode_length=episode_length,
        init_states=bool(init_states),
        gym_kwargs=gym_kwargs,
        control_mode=control_mode,
    )
    return lambda: construct_libero_eval_lane(
        base_env_fn,
        suite_name=suite_name,
        lane_idx=lane_idx,
        debug_verbose=debug_verbose,
    )


def _build_env_pool_for_lane_jobs(
    lane_jobs: list[EvalLaneJob],
    *,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: dict[str, Any],
    control_mode: str,
    dummy_env_fn: Any,
    debug_verbose: bool,
) -> AsyncVectorEnv:
    n_envs = len(lane_jobs)
    suites: dict[str, Any] = {}
    for lane_job in lane_jobs:
        suite_name = str(lane_job.task_group)
        if suite_name not in suites:
            env_module = importlib.import_module("praxis_eval.envs.libero.env")
            with suppress_libero_output(not debug_verbose):
                suites[suite_name] = env_module.get_suite(suite_name)

    env_fns = [
        _make_lane_env_fn(
            lane_idx=lane_idx,
            lane_job=lane_job,
            suite=suites[str(lane_job.task_group)],
            suite_name=str(lane_job.task_group),
            camera_names=camera_names,
            episode_length=episode_length,
            init_states=init_states,
            gym_kwargs=gym_kwargs,
            control_mode=control_mode,
            reset_stride=n_envs,
            debug_verbose=debug_verbose,
        )
        for lane_idx, lane_job in enumerate(lane_jobs)
    ]
    return AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)


def build_libero_eval_pool(
    cfg_obj: Any,
    tasks: list[tuple[str, int]],
    n_envs: int,
    debug_verbose: bool = False,
) -> EvalPoolHandle:
    """Persistent eval pool builder for LIBERO (supports multiple suites)."""
    _ = tasks
    return make_libero_eval_pool(
        cfg_obj,
        n_envs=n_envs,
        debug_verbose=debug_verbose,
    )


def _prepare_libero_eval_jobs(
    env_pool_or_handle: AsyncVectorEnv | EvalPoolHandle,
    lane_jobs: list[EvalLaneJob | None],
    *,
    camera_names: list[str] | None = None,
    episode_length: int | None = None,
    init_states: bool | None = None,
    gym_kwargs: dict[str, Any] | None = None,
    control_mode: str | None = None,
    dummy_env_fn: Any | None = None,
    debug_verbose: bool = False,
) -> None:
    active_lane_indices = [idx for idx, job in enumerate(lane_jobs) if job is not None]
    if len(active_lane_indices) == 0:
        return

    pad_job = lane_jobs[active_lane_indices[0]]
    assert pad_job is not None
    prepared_jobs: list[EvalLaneJob] = []
    for lane_job in lane_jobs:
        prepared_jobs.append(pad_job if lane_job is None else lane_job)

    if isinstance(env_pool_or_handle, EvalPoolHandle):
        if env_pool_or_handle.env_pool is None:
            if (
                camera_names is None
                or init_states is None
                or gym_kwargs is None
                or control_mode is None
                or dummy_env_fn is None
            ):
                raise RuntimeError(
                    "Lazy LIBERO eval pool initialization is missing construction context."
                )
            env_pool_or_handle.env_pool = _build_env_pool_for_lane_jobs(
                prepared_jobs,
                camera_names=camera_names,
                episode_length=episode_length,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                control_mode=control_mode,
                dummy_env_fn=dummy_env_fn,
                debug_verbose=debug_verbose,
            )
            return
        env_pool = cast(AsyncVectorEnv, env_pool_or_handle.env_pool)
    else:
        env_pool = env_pool_or_handle

    args_list = [(int(job.task_id), int(job.episode_index)) for job in prepared_jobs]
    kwargs_list = [{"task_group": str(job.task_group)} for job in prepared_jobs]
    env_pool.call_each("prepare_eval_job", args_list=args_list, kwargs_list=kwargs_list)
