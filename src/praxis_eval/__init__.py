# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone robot-policy evaluation harness."""

from importlib.metadata import PackageNotFoundError, version

from praxis_eval.api import evaluate
from praxis_eval.contracts import ActionSpec, EnvContract, ObservationKey
from praxis_eval.evaluation.config import (
    env_kwargs_without_type_task,
    env_type_from_cfg,
    optional_env_task,
    optional_env_task_ids,
    require_positive_int,
    resolve_nonnegative_int,
    resolve_optional_positive_timeout_sec,
    resolve_policy_kwargs,
)
from praxis_eval.evaluation.watchdog import (
    EvalPhaseWatchdog,
    resolve_phase_watchdog_threshold_sec,
)
from praxis_eval.policies.actions import normalize_batched_action
from praxis_eval.policies.local import LocalPolicy
from praxis_eval.registry import (
    EvalDriver,
    available_drivers,
    get_driver,
    register_driver,
)
from praxis_eval.types import (
    EvalConfig,
    EvalResult,
    EvalRuntimeHooks,
    Observation,
    Policy,
)

try:
    __version__ = version("praxis-eval")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "ActionSpec",
    "EnvContract",
    "EvalConfig",
    "EvalDriver",
    "EvalPhaseWatchdog",
    "EvalResult",
    "EvalRuntimeHooks",
    "LocalPolicy",
    "Observation",
    "ObservationKey",
    "Policy",
    "RemotePolicy",
    "__version__",
    "available_drivers",
    "build_eval_overrides_from_train_config",
    "env_kwargs_without_type_task",
    "env_type_from_cfg",
    "evaluate",
    "get_driver",
    "infer_eval_env_target",
    "normalize_batched_action",
    "optional_env_task",
    "optional_env_task_ids",
    "register_driver",
    "normalize_eval_overrides",
    "require_positive_int",
    "resolve_eval_step_dir",
    "resolve_nonnegative_int",
    "resolve_optional_positive_timeout_sec",
    "resolve_phase_watchdog_threshold_sec",
    "resolve_policy_kwargs",
]


def __getattr__(name: str):
    """Lazily expose optional remote policy support."""
    if name == "RemotePolicy":
        from praxis_eval.policies.remote import RemotePolicy

        return RemotePolicy
    if name == "infer_eval_env_target":
        from praxis_eval.envs.factory import infer_eval_env_target

        return infer_eval_env_target
    if name == "resolve_eval_step_dir":
        from praxis_eval.evaluation.artifacts import resolve_eval_step_dir

        return resolve_eval_step_dir
    if name == "build_eval_overrides_from_train_config":
        from praxis_eval.evaluation.overrides import (
            build_eval_overrides_from_train_config,
        )

        return build_eval_overrides_from_train_config
    if name == "normalize_eval_overrides":
        from praxis_eval.evaluation.overrides import normalize_eval_overrides

        return normalize_eval_overrides
    raise AttributeError(name)
