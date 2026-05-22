# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa env observation processor — mirrors LiberoProcessorStep."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import IdentityProcessorStep, PolicyProcessorPipeline
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.utils.constants import OBS_PREFIX, OBS_STATE

from praxis_eval.envs.robocasa.state import (
    ROBOCASA_STATE_PORTS,
    state_dim_from_keys,
    state_keys_from_ports,
)

# robot_state key after preprocess_observation.
_OBS_ROBOT_STATE = f"{OBS_PREFIX}robot_state"


@dataclass
class RobocasaProcessorStep(ObservationProcessorStep):
    """Process RoboCasa observations into the LeRobot standard format.

    Input (after ``preprocess_observation`` + ``add_envs_task``):
      - ``observation.images.<cam>``  (B, C, H, W) float32 in [0, 1]
      - ``observation.robot_state``   dict of tensors keyed by state sub-key
      - ``observation.task``          list[str]

    Output:
      - ``observation.images.<cam>``  unchanged — ``RobocasaEnv`` already corrects
                                      robosuite's upside-down convention
      - ``observation.state``         (B, state_dim) float32 — concatenation of
                                      state_keys in order
      - ``observation.task``          unchanged

    Args:
        state_keys: Sub-keys of ``robot_state`` to concatenate, in order.
            Defaults to the official RoboCasa v1.0 LeRobot modality:
            ``("base_pos", "base_quat", "base_to_eef_pos",
            "base_to_eef_quat", "gripper_qpos")`` → state_dim=16.
    """

    state_keys: tuple[str, ...] = field(
        default_factory=lambda: state_keys_from_ports(ROBOCASA_STATE_PORTS)
    )

    def observation(self, observation: dict) -> dict:
        processed = {k: v for k, v in observation.items() if k != _OBS_ROBOT_STATE}

        if _OBS_ROBOT_STATE in observation:
            robot_state = observation[_OBS_ROBOT_STATE]
            components: list[torch.Tensor] = []
            for key in self.state_keys:
                if key not in robot_state:
                    raise KeyError(
                        f"RobocasaProcessorStep: expected robot_state[{key!r}], "
                        f"got keys: {list(robot_state.keys())}"
                    )
                comp = robot_state[key].float()
                if comp.dim() == 1:
                    comp = comp.unsqueeze(0)
                components.append(comp)
                # require per-key observation.state.robot0_* keys in PolicyIO.
                processed[f"{OBS_PREFIX}state.robot0_{key}"] = comp

            processed[OBS_STATE] = torch.cat(components, dim=-1)

        return processed

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Replace per-key STATE entries with a single flat ``observation.state``."""
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}

        for ft, feats in features.items():
            if ft != FeatureType.STATE:
                new_features[ft] = feats.copy()

        state_dim = state_dim_from_keys(self.state_keys)
        state_ft = cast(PipelineFeatureType, FeatureType.STATE)
        new_features[state_ft] = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
        }
        return new_features


def make_robocasa_env_pre_post_processors(
    env_cfg: Any,
    policy_cfg: Any,
):
    """Create RoboCasa env processors."""
    _ = env_cfg, policy_cfg
    return (
        PolicyProcessorPipeline(steps=[RobocasaProcessorStep()]),
        PolicyProcessorPipeline(steps=[IdentityProcessorStep()]),
    )
