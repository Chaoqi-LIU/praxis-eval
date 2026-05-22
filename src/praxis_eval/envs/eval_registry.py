# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Eval-driver registry for env-family-specific evaluation orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from praxis_eval.contracts import ActionSpec
from praxis_eval.types import Policy

EnvRuntimeDriver = Callable[[dict[str, Any], Any, "EvalDriverContext"], dict[str, Any]]
_ENV_RUNTIME_DRIVER_REGISTRY: dict[str, EnvRuntimeDriver | str] = {}


@dataclass
class EvalDriverContext:
    """Shared runtime context passed to env-family eval drivers."""

    cfg: DictConfig
    seed: int
    eval_mode: str
    eval_output_dir: Path
    eval_media_dir: Path
    num_eval_per_task: int
    num_parallel_env: int
    eval_record_episodes_per_task: int
    eval_debug_verbose: bool
    eval_step_timeout_sec: float | None
    eval_policy_kwargs: dict[str, Any]
    eval_device: str | Any
    server_address: str | None
    policy: Policy | None
    policy_preprocessor: Any
    policy_postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any
    phase_heartbeat: Callable[[str], None]
    progress_heartbeat: Callable[[str], None]
    action_spec: ActionSpec | None = None
    eval_rollout_failure_retries: int = 1


def register_env_runtime_driver(name: str, driver: EnvRuntimeDriver | str) -> None:
    """Register env-family-specific runtime orchestration."""
    _ENV_RUNTIME_DRIVER_REGISTRY[_normalize_type_name(name)] = driver


def available_env_runtime_driver_types() -> tuple[str, ...]:
    """Return env families with custom runtime orchestration."""
    return tuple(sorted(_ENV_RUNTIME_DRIVER_REGISTRY))


def run_env_runtime_driver(
    env_cfg: DictConfig | dict[str, Any],
    *,
    context: EvalDriverContext,
) -> dict[str, Any]:
    """Resolve and run the appropriate evaluation driver for ``env_cfg``."""
    from praxis_eval.envs.factory import build_env_config

    raw_cfg = _to_plain_dict(env_cfg)
    cfg_obj = build_env_config(env_cfg)
    env_type = _normalize_type_name(raw_cfg.get("type", ""))
    driver = _resolve_env_runtime_driver(
        _ENV_RUNTIME_DRIVER_REGISTRY.get(env_type, _run_async_pool_eval)
    )
    return driver(raw_cfg, cfg_obj, context)


def _run_async_pool_eval(
    raw_cfg: dict[str, Any],
    _cfg_obj: Any,
    context: EvalDriverContext,
) -> dict[str, Any]:
    """Default async-pool evaluation path used by existing env families."""
    from praxis_eval.envs.factory import list_tasks, make_eval_async_pool
    from praxis_eval.evaluation.sim import evaluate_policy_on_env_pool

    tasks = list_tasks(raw_cfg, debug_verbose=context.eval_debug_verbose)
    n_tasks = len(tasks)
    print(
        f"Prepared {n_tasks} tasks for eval "
        f"(num_eval_per_task={context.num_eval_per_task}, "
        f"num_parallel_env={context.num_parallel_env}, async=True)"
    )

    if context.policy is None:
        raise RuntimeError("Eval driver requires a policy.")
    eval_policy = context.policy

    context.phase_heartbeat("before_pool_construction")
    env_pool = make_eval_async_pool(
        raw_cfg,
        tasks=tasks,
        n_envs=context.num_parallel_env,
        debug_verbose=context.eval_debug_verbose,
    )
    try:
        context.phase_heartbeat("after_pool_spaces_validated")
        return evaluate_policy_on_env_pool(
            tasks=tasks,
            eval_pool=env_pool,
            policy=eval_policy,
            num_eval_per_task=context.num_eval_per_task,
            start_seed=context.seed,
            device=context.eval_device,
            preprocessor=context.policy_preprocessor,
            postprocessor=context.policy_postprocessor,
            env_preprocessor=context.env_preprocessor,
            env_postprocessor=context.env_postprocessor,
            policy_kwargs=context.eval_policy_kwargs,
            action_spec=context.action_spec,
            rollout_step_timeout_sec=context.eval_step_timeout_sec,
            rollout_failure_retries=context.eval_rollout_failure_retries,
            max_episodes_rendered_per_task=context.eval_record_episodes_per_task,
            videos_dir=context.eval_media_dir
            if context.eval_record_episodes_per_task > 0
            else None,
            phase_heartbeat=context.phase_heartbeat,
            progress_heartbeat=context.progress_heartbeat,
        )
    finally:
        env_pool.close(terminate=True)


def _to_plain_dict(env_cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(env_cfg, DictConfig):
        from omegaconf import OmegaConf

        result = OmegaConf.to_container(env_cfg, resolve=True)
    else:
        result = dict(env_cfg)
    if not isinstance(result, dict):
        raise TypeError(f"env_cfg must resolve to dict, got {type(result)!r}")
    return {str(key): value for key, value in result.items()}


def _normalize_type_name(name: str) -> str:
    return str(name).strip().lower()


def _resolve_env_runtime_driver(
    driver: EnvRuntimeDriver | str,
) -> EnvRuntimeDriver:
    if callable(driver):
        return driver
    module_name, separator, attr_name = str(driver).partition(":")
    if separator != ":" or not module_name or not attr_name:
        raise ValueError(
            "Env runtime driver import paths must use 'module:function' format; "
            f"got {driver!r}."
        )
    loaded = getattr(import_module(module_name), attr_name)
    if not callable(loaded):
        raise TypeError(f"Env runtime driver {driver!r} is not callable.")
    return loaded


register_env_runtime_driver("simpler", "praxis_eval.envs.simpler.eval:run_simpler_eval")
register_env_runtime_driver("mshab", "praxis_eval.envs.mshab.eval:run_mshab_eval")
