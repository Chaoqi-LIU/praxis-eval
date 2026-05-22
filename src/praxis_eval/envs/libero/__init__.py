# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO env helpers owned by the LIBERO env family."""

import importlib
from typing import TYPE_CHECKING

from praxis_eval.contracts import ActionSpec, EnvContract, ObservationKey

CONTRACT = EnvContract(
    env_type="libero",
    observation_keys=(
        ObservationKey("task", "str", description="Natural language instruction."),
        ObservationKey(
            "observation.images.<camera>",
            "uint8|float32",
            shape=("C", "H", "W"),
            description="RGB camera image keyed by camera name.",
        ),
    ),
    action=ActionSpec(
        shape=(7,),
        dtype="float32",
        minimum=-1.0,
        maximum=1.0,
        convention="normalized_delta_pose_gripper",
        description="Normalized robot action expected by the LIBERO controller.",
    ),
)

if TYPE_CHECKING:
    from praxis_eval.envs.libero.config import LiberoEnvConfig
    from praxis_eval.envs.libero.env import LiberoEnv, create_libero_envs
    from praxis_eval.envs.libero.eval import make_libero_eval_pool
    from praxis_eval.envs.libero.output import suppress_libero_output
    from praxis_eval.envs.libero.runtime import (
        LiberoEvalLaneWrapper,
        construct_libero_eval_lane,
        make_dummy_libero_env_fn,
        make_libero_env_fn,
    )

__all__ = [
    "LiberoEnv",
    "LiberoEnvConfig",
    "LiberoEvalLaneWrapper",
    "CONTRACT",
    "construct_libero_eval_lane",
    "create_libero_envs",
    "infer_libero_eval_target_from_dataset",
    "list_libero_tasks",
    "make_dummy_libero_env_fn",
    "make_libero_env_fn",
    "make_libero_eval_pool",
    "suppress_libero_output",
]


def __getattr__(name: str):
    if name == "LiberoEnvConfig":
        from praxis_eval.envs.libero.config import LiberoEnvConfig

        return LiberoEnvConfig
    if name in {"LiberoEnv", "create_libero_envs"}:
        _env = importlib.import_module("praxis_eval.envs.libero.env")

        return getattr(_env, name)
    if name == "make_libero_eval_pool":
        from praxis_eval.envs.libero.eval import make_libero_eval_pool

        return make_libero_eval_pool
    if name == "suppress_libero_output":
        from praxis_eval.envs.libero.output import suppress_libero_output

        return suppress_libero_output
    if name in {"infer_libero_eval_target_from_dataset", "list_libero_tasks"}:
        _tasks = importlib.import_module("praxis_eval.envs.libero.tasks")

        return getattr(_tasks, name)
    if name in {
        "LiberoEvalLaneWrapper",
        "construct_libero_eval_lane",
        "make_dummy_libero_env_fn",
        "make_libero_env_fn",
    }:
        _runtime = importlib.import_module("praxis_eval.envs.libero.runtime")

        return getattr(_runtime, name)
    raise AttributeError(name)
