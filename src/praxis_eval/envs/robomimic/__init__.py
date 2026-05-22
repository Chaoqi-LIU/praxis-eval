# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboMimic env family with lazy heavy-runtime imports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from praxis_eval.contracts import ActionSpec, EnvContract, ObservationKey

CONTRACT = EnvContract(
    env_type="robomimic",
    observation_keys=(
        ObservationKey("task", "str", description="Task instruction."),
        ObservationKey("observation.state", "float32", shape=(9,)),
        ObservationKey(
            "observation.images.<camera>",
            "float32",
            shape=("C", "H", "W"),
        ),
    ),
    action=ActionSpec(
        shape=(7,),
        dtype="float32",
        minimum=-1.0,
        maximum=1.0,
        convention="robomimic_delta_pose_gripper",
    ),
)

if TYPE_CHECKING:
    from praxis_eval.envs.robomimic.config import RobomimicEnvConfig
    from praxis_eval.envs.robomimic.env import RobomimicEnv
    from praxis_eval.envs.robomimic.eval import build_robomimic_eval_pool
    from praxis_eval.envs.robomimic.runtime import (
        RobomimicEvalLaneWrapper,
        make_dummy_robomimic_env_fn,
    )

__all__ = [
    "RobomimicEnv",
    "RobomimicEnvConfig",
    "RobomimicEvalLaneWrapper",
    "CONTRACT",
    "build_robomimic_eval_pool",
    "infer_robomimic_eval_target_from_dataset",
    "list_robomimic_tasks",
    "make_dummy_robomimic_env_fn",
]


def __getattr__(name: str):
    if name == "RobomimicEnvConfig":
        from praxis_eval.envs.robomimic.config import RobomimicEnvConfig

        return RobomimicEnvConfig
    if name == "RobomimicEnv":
        from praxis_eval.envs.robomimic.env import RobomimicEnv

        return RobomimicEnv
    if name == "build_robomimic_eval_pool":
        from praxis_eval.envs.robomimic.eval import build_robomimic_eval_pool

        return build_robomimic_eval_pool
    if name in {"infer_robomimic_eval_target_from_dataset", "list_robomimic_tasks"}:
        _tasks = importlib.import_module("praxis_eval.envs.robomimic.tasks")

        return getattr(_tasks, name)
    if name in {"RobomimicEvalLaneWrapper", "make_dummy_robomimic_env_fn"}:
        _runtime = importlib.import_module("praxis_eval.envs.robomimic.runtime")
        return getattr(_runtime, name)
    raise AttributeError(name)
