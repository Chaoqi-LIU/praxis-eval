# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""MS-HAB registration hooks."""

from __future__ import annotations


def register_mshab_env_family() -> None:
    """Register MS-HAB with the env factory."""
    from praxis_eval.envs.factory import (
        register_env_config,
        register_eval_target_inferer,
        register_task_lister,
    )
    from praxis_eval.envs.mshab.eval import (
        infer_mshab_eval_target_from_dataset,
        list_mshab_tasks,
    )

    register_env_config("mshab", "praxis_eval.envs.mshab.eval:MshabEnvConfig")
    register_task_lister("mshab", list_mshab_tasks)
    register_eval_target_inferer("mshab", infer_mshab_eval_target_from_dataset)
