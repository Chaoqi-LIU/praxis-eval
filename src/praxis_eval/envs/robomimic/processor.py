# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboMimic env observation processor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import IdentityProcessorStep, PolicyProcessorPipeline
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.utils.constants import OBS_PREFIX, OBS_STATE

from praxis_eval.envs.robomimic.state import (
    ROBOMIMIC_STATE_PORTS,
    state_dim_from_ports,
)

_OBS_ROBOT_STATE = f"{OBS_PREFIX}robot_state"


@dataclass
class RobomimicProcessorStep(ObservationProcessorStep):
    """Flatten nested RoboMimic robot_state observations."""

    state_ports: tuple[str, ...] = field(
        default_factory=lambda: tuple(ROBOMIMIC_STATE_PORTS)
    )

    def observation(self, observation: dict) -> dict:
        processed = {
            key: value for key, value in observation.items() if key != _OBS_ROBOT_STATE
        }

        if _OBS_ROBOT_STATE in observation:
            robot_state = observation[_OBS_ROBOT_STATE]
            components: list[torch.Tensor] = []
            for port in self.state_ports:
                if port not in robot_state:
                    raise KeyError(
                        f"RobomimicProcessorStep: expected robot_state[{port!r}], "
                        f"got keys: {list(robot_state.keys())}"
                    )
                comp = robot_state[port].float()
                if comp.dim() == 1:
                    comp = comp.unsqueeze(0)
                components.append(comp)
                processed[f"{OBS_STATE}.{port}"] = comp

            processed[OBS_STATE] = torch.cat(components, dim=-1)

        return processed

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Replace per-port state entries with one flat ``observation.state``."""
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}
        for ft, feats in features.items():
            if ft != FeatureType.STATE:
                new_features[ft] = feats.copy()

        state_ft = cast(PipelineFeatureType, FeatureType.STATE)
        new_features[state_ft] = {
            OBS_STATE: PolicyFeature(
                type=FeatureType.STATE,
                shape=(state_dim_from_ports(self.state_ports),),
            )
        }
        return new_features


def make_robomimic_env_pre_post_processors(
    env_cfg: Any,
    policy_cfg: Any,
):
    """Create RoboMimic env processors."""
    _ = policy_cfg
    return (
        PolicyProcessorPipeline(
            steps=[RobomimicProcessorStep(state_ports=tuple(env_cfg.state_ports))]
        ),
        PolicyProcessorPipeline(steps=[IdentityProcessorStep()]),
    )
