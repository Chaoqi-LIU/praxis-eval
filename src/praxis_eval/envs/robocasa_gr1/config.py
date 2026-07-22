# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa GR-1 environment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION

from praxis_eval.envs.robocasa_gr1.spec import (
    GR1_ACTION_DIM,
    GR1_STATE_DIMS,
    GR1_VIDEO_KEYS,
)


def _default_features() -> dict[str, PolicyFeature]:
    features: dict[str, PolicyFeature] = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3))
        for key in GR1_VIDEO_KEYS
    }
    features.update(
        {
            key: PolicyFeature(type=FeatureType.STATE, shape=(width,))
            for key, width in GR1_STATE_DIMS.items()
        }
    )
    features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(GR1_ACTION_DIM,))
    return features


@dataclass
class RobocasaGr1EnvConfig:
    """Configuration for the official 24-task GR-1 tabletop benchmark."""

    processor_factory: ClassVar[str] = (
        "praxis_eval.envs.robocasa_gr1.processor:"
        "make_robocasa_gr1_env_pre_post_processors"
    )

    type: str = "robocasa_gr1"
    task: str = "all"
    task_ids: list[int] | None = None
    max_episode_steps: int = 720
    features: dict[str, PolicyFeature] = field(default_factory=_default_features)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1.")
        if not self.features_map:
            self.features_map = {key: key for key in self.features}
