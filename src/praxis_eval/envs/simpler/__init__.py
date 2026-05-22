# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""SimplerEnv contract."""

from praxis_eval.contracts import ActionSpec, EnvContract, ObservationKey
from praxis_eval.envs.simpler.eval import SimplerEnvConfig, SimplerTaskSpec

CONTRACT = EnvContract(
    env_type="simpler",
    observation_keys=(
        ObservationKey("task", "str", description="Bridge task instruction."),
        ObservationKey(
            "observation.images.image",
            "uint8|float32",
            shape=("C", "H", "W"),
        ),
        ObservationKey("observation.state", "float32", description="Optional state."),
    ),
    action=ActionSpec(
        shape=(7,),
        dtype="float32",
        convention="simpler_bridge_widowx_action",
    ),
)

__all__ = ["CONTRACT", "SimplerEnvConfig", "SimplerTaskSpec"]
