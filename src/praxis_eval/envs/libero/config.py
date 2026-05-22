# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO config bootstrap owned by the LIBERO env family."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import (
    ACTION,
    LIBERO_KEY_EEF_MAT,
    LIBERO_KEY_EEF_POS,
    LIBERO_KEY_EEF_QUAT,
    LIBERO_KEY_GRIPPER_QPOS,
    LIBERO_KEY_GRIPPER_QVEL,
    LIBERO_KEY_JOINTS_POS,
    LIBERO_KEY_JOINTS_VEL,
    OBS_IMAGES,
    OBS_STATE,
)

from praxis_eval.envs.libero.spec import (
    ACTION_DIM,
    DEFAULT_CAMERA_NAME,
    parse_camera_names,
    resolve_camera_name_mapping,
)


def _libero_pixel_feature_key(camera_name: str) -> str:
    return f"pixels/{camera_name}"


def _default_features(
    *,
    obs_type: str,
    camera_name: str | list[str] | tuple[str, ...],
    observation_height: int,
    observation_width: int,
) -> dict[str, PolicyFeature]:
    features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    for camera in parse_camera_names(camera_name):
        features[_libero_pixel_feature_key(camera)] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(int(observation_height), int(observation_width), 3),
        )
    if obs_type == "pixels_agent_pos":
        features.update(
            {
                LIBERO_KEY_EEF_POS: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
                LIBERO_KEY_EEF_QUAT: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
                LIBERO_KEY_EEF_MAT: PolicyFeature(type=FeatureType.STATE, shape=(3, 3)),
                LIBERO_KEY_GRIPPER_QPOS: PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(2,),
                ),
                LIBERO_KEY_GRIPPER_QVEL: PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(2,),
                ),
                LIBERO_KEY_JOINTS_POS: PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(7,),
                ),
                LIBERO_KEY_JOINTS_VEL: PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(7,),
                ),
            }
        )
    elif obs_type != "pixels":
        raise ValueError(f"Unsupported LIBERO obs_type: {obs_type!r}.")
    return features


def _default_features_map(
    *,
    obs_type: str,
    camera_name: str | list[str] | tuple[str, ...],
    camera_name_mapping: dict[str, str] | None,
) -> dict[str, str]:
    camera_names = parse_camera_names(camera_name)
    mapping = resolve_camera_name_mapping(camera_names, camera_name_mapping)
    features_map = {
        ACTION: ACTION,
    }
    if obs_type == "pixels_agent_pos":
        features_map.update(
            {
                LIBERO_KEY_EEF_POS: f"{OBS_STATE}.eef_pos",
                LIBERO_KEY_EEF_QUAT: f"{OBS_STATE}.eef_quat",
                LIBERO_KEY_EEF_MAT: f"{OBS_STATE}.eef_mat",
                LIBERO_KEY_GRIPPER_QPOS: f"{OBS_STATE}.gripper_qpos",
                LIBERO_KEY_GRIPPER_QVEL: f"{OBS_STATE}.gripper_qvel",
                LIBERO_KEY_JOINTS_POS: f"{OBS_STATE}.joint_pos",
                LIBERO_KEY_JOINTS_VEL: f"{OBS_STATE}.joint_vel",
            }
        )
    for camera, output_name in mapping.items():
        features_map[_libero_pixel_feature_key(camera)] = f"{OBS_IMAGES}.{output_name}"
    return features_map


@dataclass
class LiberoEnvConfig:
    """Configuration for evaluator-owned LIBERO environments."""

    processor_factory: ClassVar[str] = (
        "praxis_eval.envs.libero.processor:make_libero_env_pre_post_processors"
    )

    type: str = "libero"
    task: str = "libero_10"
    task_ids: list[int] | None = None
    fps: int = 30
    episode_length: int | None = None
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = DEFAULT_CAMERA_NAME
    init_states: bool = True
    camera_name_mapping: dict[str, str] | None = None
    observation_height: int = 360
    observation_width: int = 360
    visualization_height: int = 480
    visualization_width: int = 640
    control_mode: str = "relative"
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.obs_type not in {"pixels", "pixels_agent_pos"}:
            raise ValueError(f"Unsupported LIBERO obs_type: {self.obs_type!r}.")
        if self.control_mode not in {"relative", "absolute"}:
            raise ValueError(f"Unsupported LIBERO control_mode: {self.control_mode!r}.")
        camera_names = parse_camera_names(self.camera_name)
        self.camera_name_mapping = resolve_camera_name_mapping(
            camera_names,
            self.camera_name_mapping,
        )
        if not self.features:
            self.features = _default_features(
                obs_type=self.obs_type,
                camera_name=self.camera_name,
                observation_height=int(self.observation_height),
                observation_width=int(self.observation_width),
            )
        if not self.features_map:
            self.features_map = _default_features_map(
                obs_type=self.obs_type,
                camera_name=self.camera_name,
                camera_name_mapping=self.camera_name_mapping,
            )

    @property
    def gym_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_height": int(self.visualization_height),
            "visualization_width": int(self.visualization_width),
        }
        if self.task_ids is not None:
            kwargs["task_ids"] = list(self.task_ids)
        return kwargs


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_config_root() -> Path:
    return _repo_root() / ".tmp" / "libero_config"


def _installed_benchmark_root() -> Path:
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            "Could not locate the installed `libero.libero` package."
        )
    return Path(spec.origin).resolve().parent


def _desired_config(benchmark_root: Path) -> dict[str, str]:
    return {
        "benchmark_root": benchmark_root.as_posix(),
        "bddl_files": (benchmark_root / "bddl_files").as_posix(),
        "init_states": (benchmark_root / "init_files").as_posix(),
        "datasets": (benchmark_root.parent / "datasets").as_posix(),
        "assets": (benchmark_root / "assets").as_posix(),
    }


def ensure_libero_config(*, config_root: Path | None = None) -> Path:
    """Point LIBERO at an evaluator-owned config rooted at the installed package."""

    config_root = (
        Path(config_root).expanduser().resolve()
        if config_root is not None
        else _default_config_root().resolve()
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)

    benchmark_root = _installed_benchmark_root()
    config = _desired_config(benchmark_root)

    config_root.mkdir(parents=True, exist_ok=True)
    config_file = config_root / "config.yaml"

    existing: dict[str, str] | None = None
    if config_file.exists():
        with config_file.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            existing = {str(key): str(value) for key, value in loaded.items()}

    if existing != config:
        with config_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=True)

    return config_file
