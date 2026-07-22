# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Register the in-process RoboCasa GR-1 benchmark family."""

from __future__ import annotations


def register_robocasa_gr1_env_family() -> None:
    from praxis_eval.envs.factory import (
        register_env_config,
        register_eval_pool_builder,
        register_eval_target_inferer,
        register_task_lister,
    )

    register_env_config(
        "robocasa_gr1",
        "praxis_eval.envs.robocasa_gr1.config:RobocasaGr1EnvConfig",
    )
    register_task_lister(
        "robocasa_gr1",
        "praxis_eval.envs.robocasa_gr1.tasks:list_robocasa_gr1_tasks",
    )
    register_eval_target_inferer(
        "robocasa_gr1",
        "praxis_eval.envs.robocasa_gr1.tasks:"
        "infer_robocasa_gr1_eval_target_from_dataset",
    )
    register_eval_pool_builder(
        "robocasa_gr1",
        "praxis_eval.envs.robocasa_gr1.eval:build_robocasa_gr1_eval_pool",
    )
