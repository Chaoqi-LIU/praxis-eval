# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa env family — lazy imports to avoid triggering OpenGL at collection time."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from praxis_eval.contracts import ActionSpec, EnvContract, ObservationKey

CONTRACT = EnvContract(
    env_type="robocasa",
    observation_keys=(
        ObservationKey("task", "str", description="Episode language instruction."),
        ObservationKey("observation.state", "float32", shape=(16,)),
        ObservationKey(
            "observation.images.<camera>",
            "float32",
            shape=("C", "H", "W"),
        ),
    ),
    action=ActionSpec(
        shape=(12,),
        dtype="float32",
        minimum=-1.0,
        maximum=1.0,
        convention="robocasa_normalized_mobile_manipulator_action",
    ),
)

if TYPE_CHECKING:
    from praxis_eval.envs.robocasa.config import RobocasaEnvConfig
    from praxis_eval.envs.robocasa.env import RobocasaEnv
    from praxis_eval.envs.robocasa.eval import build_robocasa_eval_pool
    from praxis_eval.envs.robocasa.runtime import (
        RobocasaEvalLaneWrapper,
        make_dummy_robocasa_env_fn,
    )

__all__ = [
    "RobocasaEnv",
    "RobocasaEnvConfig",
    "RobocasaEvalLaneWrapper",
    "CONTRACT",
    "build_robocasa_eval_pool",
    "infer_robocasa_eval_target_from_dataset",
    "list_robocasa_tasks",
    "make_dummy_robocasa_env_fn",
]


def __getattr__(name: str):
    if name == "RobocasaEnvConfig":
        from praxis_eval.envs.robocasa.config import RobocasaEnvConfig

        return RobocasaEnvConfig
    if name == "RobocasaEnv":
        from praxis_eval.envs.robocasa.env import RobocasaEnv

        return RobocasaEnv
    if name == "build_robocasa_eval_pool":
        from praxis_eval.envs.robocasa.eval import build_robocasa_eval_pool

        return build_robocasa_eval_pool
    if name in {"infer_robocasa_eval_target_from_dataset", "list_robocasa_tasks"}:
        _tasks = importlib.import_module("praxis_eval.envs.robocasa.tasks")

        return getattr(_tasks, name)
    if name in {"RobocasaEvalLaneWrapper", "make_dummy_robocasa_env_fn"}:
        _runtime = importlib.import_module("praxis_eval.envs.robocasa.runtime")
        return getattr(_runtime, name)
    raise AttributeError(name)
