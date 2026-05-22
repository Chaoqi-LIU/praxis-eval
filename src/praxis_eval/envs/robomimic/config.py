# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboMimic environment configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from praxis_eval.envs.robomimic.state import ROBOMIMIC_STATE_PORTS, state_shape_for_port


class _AutoFeatures(dict[str, PolicyFeature]):
    """Sentinel dict used to detect omitted feature metadata."""


class _AutoFeaturesMap(dict[str, str]):
    """Sentinel dict used to detect omitted feature mapping."""


def _auto_features() -> dict[str, PolicyFeature]:
    return _AutoFeatures()


def _auto_features_map() -> dict[str, str]:
    return _AutoFeaturesMap()


def _default_features(
    camera_names: list[str] | None = None,
    state_ports: list[str] | None = None,
    image_size: int = 128,
) -> dict[str, PolicyFeature]:
    if camera_names is None:
        camera_names = ["agentview", "robot0_eye_in_hand"]
    if state_ports is None:
        state_ports = list(ROBOMIMIC_STATE_PORTS)

    feats: dict[str, PolicyFeature] = {}
    for cam in camera_names:
        feats[f"{OBS_IMAGES}.{cam}"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(image_size, image_size, 3),
        )
    for port in state_ports:
        feats[f"{OBS_STATE}.{port}"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=state_shape_for_port(port),
        )
    feats[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(7,))
    return feats


def _default_features_map(
    camera_names: list[str] | None = None,
    state_ports: list[str] | None = None,
) -> dict[str, str]:
    if camera_names is None:
        camera_names = ["agentview", "robot0_eye_in_hand"]
    if state_ports is None:
        state_ports = list(ROBOMIMIC_STATE_PORTS)

    mapping: dict[str, str] = {}
    for cam in camera_names:
        key = f"{OBS_IMAGES}.{cam}"
        mapping[key] = key
    for port in state_ports:
        key = f"{OBS_STATE}.{port}"
        mapping[key] = key
    mapping[ACTION] = ACTION
    return mapping


@dataclass
class RobomimicEnvConfig:
    """Configuration for a RoboMimic robosuite environment."""

    processor_factory: ClassVar[str] = (
        "praxis_eval.envs.robomimic.processor:make_robomimic_env_pre_post_processors"
    )

    type: str = "robomimic"
    task: str = "mt3"
    image_size: int = 128
    camera_names: list[str] = field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
    )
    state_ports: list[str] = field(default_factory=lambda: list(ROBOMIMIC_STATE_PORTS))
    video_camera: str = "agentview"
    video_resolution: int = 512
    max_episode_steps: int = 800
    robot: str = "Panda"
    features: dict[str, PolicyFeature] = field(default_factory=_auto_features)
    features_map: dict[str, str] = field(default_factory=_auto_features_map)

    def __post_init__(self) -> None:
        if isinstance(self.features, _AutoFeatures):
            self.features = _default_features(
                camera_names=list(self.camera_names),
                state_ports=list(self.state_ports),
                image_size=int(self.image_size),
            )
        if isinstance(self.features_map, _AutoFeaturesMap):
            self.features_map = _default_features_map(
                camera_names=list(self.camera_names),
                state_ports=list(self.state_ports),
            )
