# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa GR-1 tabletop benchmark family."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from praxis_eval.contracts import ActionSpec, EnvContract, ObservationKey
from praxis_eval.envs.robocasa_gr1.spec import GR1_ACTION_DIM

CONTRACT = EnvContract(
    env_type="robocasa_gr1",
    observation_keys=(
        ObservationKey(
            "video.ego_view_pad_res256_freq20",
            "float32",
            shape=(3, 256, 256),
        ),
        ObservationKey(
            "video.ego_view_bg_crop_pad_res256_freq20",
            "float32",
            shape=(3, 256, 256),
        ),
        ObservationKey("state.left_arm", "float32", shape=(7,)),
        ObservationKey("state.right_arm", "float32", shape=(7,)),
        ObservationKey("state.left_hand", "float32", shape=(6,)),
        ObservationKey("state.right_hand", "float32", shape=(6,)),
        ObservationKey("state.waist", "float32", shape=(3,)),
        ObservationKey("annotation.human.coarse_action", "str"),
        ObservationKey("task", "str"),
    ),
    action=ActionSpec(
        shape=(GR1_ACTION_DIM,),
        dtype="float32",
        convention=(
            "absolute_joint_positions:left_arm,right_arm,left_hand,right_hand,waist"
        ),
        description=(
            "One physical-unit action step. Policy-side action chunks must be "
            "queued and emitted one step at a time."
        ),
    ),
    notes="Runs in-process in an independent robocasa_gr1 Python namespace.",
)

if TYPE_CHECKING:
    from praxis_eval.envs.robocasa_gr1.config import RobocasaGr1EnvConfig
    from praxis_eval.envs.robocasa_gr1.env import RobocasaGr1Env
    from praxis_eval.envs.robocasa_gr1.eval import build_robocasa_gr1_eval_pool

__all__ = [
    "CONTRACT",
    "RobocasaGr1Env",
    "RobocasaGr1EnvConfig",
    "build_robocasa_gr1_eval_pool",
    "infer_robocasa_gr1_eval_target_from_dataset",
    "list_robocasa_gr1_tasks",
]


def __getattr__(name: str):
    modules = {
        "RobocasaGr1EnvConfig": "praxis_eval.envs.robocasa_gr1.config",
        "RobocasaGr1Env": "praxis_eval.envs.robocasa_gr1.env",
        "build_robocasa_gr1_eval_pool": "praxis_eval.envs.robocasa_gr1.eval",
        "list_robocasa_gr1_tasks": "praxis_eval.envs.robocasa_gr1.tasks",
        "infer_robocasa_gr1_eval_target_from_dataset": (
            "praxis_eval.envs.robocasa_gr1.tasks"
        ),
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(importlib.import_module(module_name), name)
