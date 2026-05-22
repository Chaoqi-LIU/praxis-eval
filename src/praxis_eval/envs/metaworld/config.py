# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""MetaWorld environment configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

from praxis_eval.envs.metaworld.spec import resolve_episode_length


def _default_features(
    observation_height: int = 480,
    observation_width: int = 480,
) -> dict[str, PolicyFeature]:
    return {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        "pixels/top": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(int(observation_height), int(observation_width), 3),
        ),
    }


def _default_features_map() -> dict[str, str]:
    return {
        ACTION: ACTION,
        "agent_pos": OBS_STATE,
        "pixels/top": OBS_IMAGE,
    }


@dataclass
class MetaworldEnvConfig:
    """Configuration for evaluator-owned MetaWorld environments."""

    processor_factory: ClassVar[str] = "identity"

    type: str = "metaworld"
    task: str = "mt50"
    fps: int = 80
    episode_length: int | None = None
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "corner2"
    observation_height: int = 480
    observation_width: int = 480
    visualization_height: int = 480
    visualization_width: int = 640
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=_default_features_map)

    def __post_init__(self) -> None:
        if self.obs_type not in {"pixels", "pixels_agent_pos"}:
            raise ValueError(f"Unsupported MetaWorld obs_type: {self.obs_type!r}.")
        if self.episode_length is not None:
            self.episode_length = resolve_episode_length(self.episode_length)
        if not self.features:
            self.features = _default_features(
                observation_height=int(self.observation_height),
                observation_width=int(self.observation_width),
            )
            if self.obs_type == "pixels":
                self.features.pop("agent_pos", None)
                self.features_map.pop("agent_pos", None)

    @property
    def gym_kwargs(self) -> dict[str, Any]:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "camera_name": self.camera_name,
            "observation_height": int(self.observation_height),
            "observation_width": int(self.observation_width),
            "visualization_height": int(self.visualization_height),
            "visualization_width": int(self.visualization_width),
        }
