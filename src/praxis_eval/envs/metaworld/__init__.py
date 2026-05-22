# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""MetaWorld env helpers owned by the MetaWorld env family."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from praxis_eval.contracts import ActionSpec, EnvContract, ObservationKey

CONTRACT = EnvContract(
    env_type="metaworld",
    observation_keys=(
        ObservationKey("task", "str", description="Task instruction or task id."),
        ObservationKey("observation.state", "float32", description="Proprio state."),
        ObservationKey(
            "observation.images.<camera>",
            "uint8|float32",
            shape=("C", "H", "W"),
            description="Optional RGB camera image.",
        ),
    ),
    action=ActionSpec(
        shape=(4,),
        dtype="float32",
        minimum=-1.0,
        maximum=1.0,
        convention="metaworld_xyz_gripper",
    ),
)

if TYPE_CHECKING:
    from praxis_eval.envs.metaworld.config import MetaworldEnvConfig
    from praxis_eval.envs.metaworld.env import MetaworldEnv
    from praxis_eval.envs.metaworld.eval import build_metaworld_eval_pool
    from praxis_eval.envs.metaworld.runtime import (
        MetaworldEvalLaneWrapper,
        construct_metaworld_eval_lane,
        make_dummy_metaworld_env_fn,
        make_metaworld_env_fn,
    )

__all__ = [
    "MetaworldEnvConfig",
    "MetaworldEnv",
    "MetaworldEvalLaneWrapper",
    "CONTRACT",
    "build_metaworld_eval_pool",
    "construct_metaworld_eval_lane",
    "expand_task_selectors",
    "get_task_description",
    "infer_metaworld_eval_target_from_dataset",
    "list_metaworld_tasks",
    "make_dummy_metaworld_env_fn",
    "make_metaworld_env_fn",
    "resolve_task_name",
]

_EXPORT_TO_MODULE = {
    "expand_task_selectors": "praxis_eval.envs.metaworld.tasks",
    "get_task_description": "praxis_eval.envs.metaworld.tasks",
    "infer_metaworld_eval_target_from_dataset": "praxis_eval.envs.metaworld.tasks",
    "list_metaworld_tasks": "praxis_eval.envs.metaworld.tasks",
    "resolve_task_name": "praxis_eval.envs.metaworld.tasks",
    "MetaworldEnvConfig": "praxis_eval.envs.metaworld.config",
    "MetaworldEnv": "praxis_eval.envs.metaworld.env",
    "build_metaworld_eval_pool": "praxis_eval.envs.metaworld.eval",
    "MetaworldEvalLaneWrapper": "praxis_eval.envs.metaworld.runtime",
    "construct_metaworld_eval_lane": "praxis_eval.envs.metaworld.runtime",
    "make_dummy_metaworld_env_fn": "praxis_eval.envs.metaworld.runtime",
    "make_metaworld_env_fn": "praxis_eval.envs.metaworld.runtime",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
