# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Map LeRobot's generic env observation form to official GR00T keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import IdentityProcessorStep, PolicyProcessorPipeline
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX

from praxis_eval.envs.robocasa_gr1.spec import (
    GR1_LANGUAGE_KEY,
    GR1_STATE_KEYS,
    GR1_VIDEO_KEYS,
)

_OBS_ROBOT_STATE = f"{OBS_PREFIX}robot_state"


@dataclass
class RobocasaGr1ProcessorStep(ObservationProcessorStep):
    """Expose the simulator's state, video, and language keys unchanged."""

    def observation(self, observation: dict) -> dict:
        processed = {
            key: value
            for key, value in observation.items()
            if key != _OBS_ROBOT_STATE
            and not any(key == f"{OBS_IMAGES}.{video}" for video in GR1_VIDEO_KEYS)
        }

        robot_state = observation.get(_OBS_ROBOT_STATE)
        if robot_state is not None:
            for key in GR1_STATE_KEYS:
                if key not in robot_state:
                    raise KeyError(
                        f"RoboCasa GR-1 expected robot_state[{key!r}], "
                        f"got {list(robot_state)}."
                    )
                processed[key] = robot_state[key].float()

        for key in GR1_VIDEO_KEYS:
            generic_key = f"{OBS_IMAGES}.{key}"
            if generic_key in observation:
                processed[key] = observation[generic_key]

        task = observation.get("task")
        if task is not None:
            processed[GR1_LANGUAGE_KEY] = task
        return processed

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """The config already declares policy-facing official GR-1 keys."""
        return {
            feature_type: values.copy() for feature_type, values in features.items()
        }


def make_robocasa_gr1_env_pre_post_processors(
    env_cfg: Any,
    policy_cfg: Any,
):
    """Create the GR-1 observation adapter and identity action adapter."""
    _ = env_cfg, policy_cfg
    return (
        PolicyProcessorPipeline(steps=[RobocasaGr1ProcessorStep()]),
        PolicyProcessorPipeline(steps=[IdentityProcessorStep()]),
    )
