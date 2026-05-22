# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""SimplerEnv registration hooks."""

from __future__ import annotations


def register_simpler_env_family() -> None:
    """Register SimplerEnv with the env factory."""
    from praxis_eval.envs.factory import (
        register_env_config,
        register_eval_target_inferer,
        register_task_lister,
    )
    from praxis_eval.envs.simpler.eval import (
        infer_simpler_eval_target_from_dataset,
        list_simpler_tasks,
    )

    register_env_config("simpler", "praxis_eval.envs.simpler.eval:SimplerEnvConfig")
    register_task_lister("simpler", list_simpler_tasks)
    register_eval_target_inferer("simpler", infer_simpler_eval_target_from_dataset)
