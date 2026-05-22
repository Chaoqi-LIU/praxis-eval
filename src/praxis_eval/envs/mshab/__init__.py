# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""MS-HAB contract."""

from praxis_eval.contracts import ActionSpec, EnvContract, ObservationKey
from praxis_eval.envs.mshab.eval import MshabEnvConfig, MshabTaskSpec

CONTRACT = EnvContract(
    env_type="mshab",
    observation_keys=(
        ObservationKey("task", "str", description="MS-HAB task instruction."),
        ObservationKey("observation.state", "float32", shape=(42,)),
        ObservationKey(
            "observation.images.fetch_head",
            "float32",
            shape=(3, 128, 128),
        ),
        ObservationKey(
            "observation.images.fetch_hand",
            "float32",
            shape=(3, 128, 128),
        ),
    ),
    action=ActionSpec(
        shape=(13,),
        dtype="float32",
        minimum=-1.0,
        maximum=1.0,
        convention="mshab_normalized_controller_action",
    ),
)

__all__ = ["CONTRACT", "MshabEnvConfig", "MshabTaskSpec"]
