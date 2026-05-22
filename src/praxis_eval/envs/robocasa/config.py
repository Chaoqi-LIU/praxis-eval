# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa environment configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from praxis_eval.envs.robocasa.state import ROBOCASA_STATE_PORTS, state_shape_for_port

# Dataset feature keys written by prepare_robocasa_dataset.py.
# images: observation.images.<cam_name>
# state:  observation.state built from the official RoboCasa v1.0 modality:
#         base_pos + base_quat + base_to_eef_pos + base_to_eef_quat + gripper_qpos
# action: action


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
    image_size: int = 128,
) -> dict[str, PolicyFeature]:
    if camera_names is None:
        camera_names = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]
    feats: dict[str, PolicyFeature] = {}
    for cam in camera_names:
        feats[f"{OBS_IMAGES}.{cam}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(image_size, image_size, 3)
        )
    # The env emits robot_state dict entries, then RobocasaProcessorStep collapses
    # them into the flat observation.state key used by the dataset and checkpoints.
    for port in ROBOCASA_STATE_PORTS:
        feats[f"{OBS_STATE}.{port}"] = PolicyFeature(
            type=FeatureType.STATE, shape=state_shape_for_port(port)
        )
    feats[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(12,))
    return feats


def _default_features_map(
    camera_names: list[str] | None = None,
) -> dict[str, str]:
    """Map dataset keys → policy input keys (identity for images; state kept as-is
    until ``RobocasaProcessorStep.transform_features`` collapses them)."""
    if camera_names is None:
        camera_names = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]
    mapping: dict[str, str] = {}
    for cam in camera_names:
        key = f"{OBS_IMAGES}.{cam}"
        mapping[key] = key
    for port in ROBOCASA_STATE_PORTS:
        key = f"{OBS_STATE}.{port}"
        mapping[key] = key
    mapping[ACTION] = ACTION
    return mapping


@dataclass
class RobocasaEnvConfig:
    """Configuration for a RoboCasa environment.

    ``task`` can be a single RoboCasa365 leaf task (e.g. ``"CloseDrawer"``)
    or a evaluator multi-task group key such as ``"mt5"``.
    """

    processor_factory: ClassVar[str] = (
        "praxis_eval.envs.robocasa.processor:make_robocasa_env_pre_post_processors"
    )

    type: str = "robocasa"
    task: str = "mt5"
    split: Literal["all", "pretrain", "target"] = "all"
    image_size: int = 128
    camera_names: list[str] = field(
        default_factory=lambda: [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]
    )
    max_episode_steps: int = 500
    features: dict[str, PolicyFeature] = field(default_factory=_auto_features)
    features_map: dict[str, str] = field(default_factory=_auto_features_map)

    def __post_init__(self) -> None:
        if isinstance(self.features, _AutoFeatures):
            self.features = _default_features(
                camera_names=list(self.camera_names),
                image_size=int(self.image_size),
            )
        if isinstance(self.features_map, _AutoFeaturesMap):
            self.features_map = _default_features_map(
                camera_names=list(self.camera_names),
            )
