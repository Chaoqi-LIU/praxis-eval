# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from praxis_eval.evaluation.artifacts import (
    resolve_eval_artifact_paths,
    resolve_eval_step_dir,
    write_eval_results_json,
)

if TYPE_CHECKING:
    from praxis_eval.evaluation.sim import (
        LocalPolicyAdapter,
        evaluate_policy_on_env_pool,
    )

_SIM_EXPORTS = {
    "LocalPolicyAdapter",
    "evaluate_policy_on_env_pool",
}


def __getattr__(name: str):
    """Lazily expose sim helpers without importing env runtimes on package import."""
    if name not in _SIM_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from praxis_eval.evaluation import sim

    return getattr(sim, name)


__all__ = [
    "LocalPolicyAdapter",
    "evaluate_policy_on_env_pool",
    "resolve_eval_artifact_paths",
    "resolve_eval_step_dir",
    "write_eval_results_json",
]
