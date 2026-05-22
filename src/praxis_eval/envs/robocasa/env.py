# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa gymnasium environment wrapper.

Ported from Chaoqi-LIU/sim_env policy_research/env/robocasa/env.py and adapted
to match the evaluator's LeRobot-compatible conventions.

Observation format mirrors LIBERO so it passes through ``preprocess_observation``
unchanged:

  - ``"pixels"``: dict of ``{cam_name: HxWx3 uint8}`` for each camera
  - ``"robot_state"``: dict keyed by the official RoboCasa365 state ports.
    The default is the official RoboCasa v1.0 LeRobot modality:
    ``base_pos`` + ``base_quat`` + ``base_to_eef_pos`` + ``base_to_eef_quat`` +
    ``gripper_qpos`` for a flat 16-D proprio vector after preprocessing.

``task_description`` is exposed as a property (not an obs key) for
``add_envs_task`` compatibility.
"""

from __future__ import annotations

import inspect
import logging
import os
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Any, Literal, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from praxis_eval.envs.robocasa.runtime import _RETRYABLE_LAYOUT_ERROR_FRAGMENTS
from praxis_eval.envs.robocasa.state import ROBOCASA_STATE_PORTS, state_key_for_port

_DEFAULT_MAX_OBJAVERSE_VISUAL_MESH_BYTES = 25 * 1024 * 1024
_PATCH_SENTINEL_ATTR = "_praxis_objaverse_mesh_guard_installed"
_VALID_SCENE_SPLITS = {"all", "pretrain", "target"}
_ROBOCASA_LEROBOT_ACTION_DIM = 12
_MAX_NATIVE_ACTION_CLIP_WARNINGS = 5
_logger = logging.getLogger(__name__)
# Official RoboCasa365 LeRobot modality order:
#   base_motion, control_mode, end_effector_position,
#   end_effector_rotation, gripper_close
# Native robocasa / robosuite env action order:
#   end_effector_position, end_effector_rotation, gripper_close,
#   base_motion, control_mode
_LEROBOT_ACTION_COMPONENT_SLICES: dict[str, slice] = {
    "base_motion": slice(0, 4),
    "control_mode": slice(4, 5),
    "end_effector_position": slice(5, 8),
    "end_effector_rotation": slice(8, 11),
    "gripper_close": slice(11, 12),
}
_NATIVE_ACTION_COMPONENT_ORDER: tuple[str, ...] = (
    "end_effector_position",
    "end_effector_rotation",
    "gripper_close",
    "base_motion",
    "control_mode",
)


def _neutralize_oversized_objaverse_visual_meshes(
    xml_str: str,
    *,
    max_mesh_bytes: int,
) -> tuple[str, list[str]]:
    if max_mesh_bytes <= 0:
        return xml_str, []

    root = ET.fromstring(xml_str)
    asset = root.find("asset")
    if asset is None:
        return xml_str, []

    oversized_mesh_names: set[str] = set()
    for mesh_elem in list(asset.findall("mesh")):
        mesh_name = str(mesh_elem.get("name", ""))
        mesh_file = str(mesh_elem.get("file", ""))
        if mesh_name == "" or mesh_file == "":
            continue
        if "/models/assets/objects/objaverse/" not in mesh_file:
            continue
        if "/visual/" not in mesh_file:
            continue
        try:
            file_size = os.path.getsize(mesh_file)
        except OSError:
            continue
        if file_size <= max_mesh_bytes:
            continue
        oversized_mesh_names.add(mesh_name)
        asset.remove(mesh_elem)

    if not oversized_mesh_names:
        return xml_str, []

    for geom_elem in root.iter("geom"):
        mesh_name = geom_elem.get("mesh")
        if mesh_name is None or str(mesh_name) not in oversized_mesh_names:
            continue
        geom_elem.attrib.pop("mesh", None)
        geom_elem.set("type", "box")
        geom_elem.set("size", "0.0001 0.0001 0.0001")
        geom_elem.set("contype", "0")
        geom_elem.set("conaffinity", "0")

    return ET.tostring(root, encoding="unicode"), sorted(oversized_mesh_names)


@lru_cache(maxsize=1)
def _requires_legacy_mesh_inertia_keyword() -> bool:
    """Return whether this MuJoCo build rejects mesh inertia='shell'."""
    try:
        import mujoco

        version_raw = str(getattr(mujoco, "__version__", "0.0.0"))
        parts = version_raw.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor) < (3, 5)
    except Exception:
        return False


def _rewrite_shell_mesh_inertia_to_legacy(xml_str: str) -> tuple[str, int]:
    """Rewrite mesh inertia='shell' -> 'legacy' for older MuJoCo versions."""
    root = ET.fromstring(xml_str)
    rewritten = 0
    for mesh_elem in root.findall(".//asset/mesh"):
        if mesh_elem.get("inertia") == "shell":
            mesh_elem.set("inertia", "legacy")
            rewritten += 1
    return ET.tostring(root, encoding="unicode"), rewritten


def _install_objaverse_visual_mesh_guard() -> None:
    # Import lazily to avoid heavy robocasa import during unrelated tests.
    try:
        from robocasa.environments.kitchen.kitchen import Kitchen
    except ModuleNotFoundError:
        return

    if getattr(Kitchen, _PATCH_SENTINEL_ATTR, False):
        return

    original_edit_model_xml = Kitchen.edit_model_xml

    def _patched_edit_model_xml(self: Any, xml_str: str) -> str:
        xml_out = original_edit_model_xml(self, xml_str)
        if _requires_legacy_mesh_inertia_keyword():
            try:
                xml_out, rewritten = _rewrite_shell_mesh_inertia_to_legacy(xml_out)
            except Exception:
                _logger.exception(
                    "Failed to rewrite shell mesh inertia keywords for legacy MuJoCo."
                )
            else:
                if rewritten > 0:
                    _logger.warning(
                        "Rewrote %d mesh inertia keywords shell->legacy for legacy MuJoCo.",
                        rewritten,
                    )
        max_mesh_bytes = _DEFAULT_MAX_OBJAVERSE_VISUAL_MESH_BYTES
        if max_mesh_bytes <= 0:
            return xml_out
        try:
            patched_xml, neutralized_meshes = (
                _neutralize_oversized_objaverse_visual_meshes(
                    xml_out,
                    max_mesh_bytes=max_mesh_bytes,
                )
            )
        except Exception:
            _logger.exception(
                "Failed to apply objaverse visual mesh guard; continuing with original XML."
            )
            return xml_out
        if neutralized_meshes:
            _logger.warning(
                "Neutralized %d oversized objaverse visual meshes (max=%d bytes): %s",
                len(neutralized_meshes),
                max_mesh_bytes,
                ", ".join(neutralized_meshes[:5]),
            )
        return patched_xml

    Kitchen.edit_model_xml = _patched_edit_model_xml
    setattr(Kitchen, _PATCH_SENTINEL_ATTR, True)


def _lerobot_action_to_native_order(action: np.ndarray) -> np.ndarray:
    """Map official RoboCasa365 LeRobot action order back to env-native order."""
    action_np = np.asarray(action, dtype=np.float32)
    if action_np.shape != (_ROBOCASA_LEROBOT_ACTION_DIM,):
        raise ValueError(
            "RoboCasa expects one LeRobot-order action with shape "
            f"({_ROBOCASA_LEROBOT_ACTION_DIM},), got {tuple(action_np.shape)}."
        )
    if not np.isfinite(action_np).all():
        raise ValueError("RoboCasa action contains non-finite values.")

    native = np.empty_like(action_np, dtype=np.float32)
    offset = 0
    for component in _NATIVE_ACTION_COMPONENT_ORDER:
        src = _LEROBOT_ACTION_COMPONENT_SLICES[component]
        width = src.stop - src.start
        native[..., offset : offset + width] = action_np[..., src]
        offset += width
    return native


def _native_action_dim(env: Any) -> int:
    try:
        action_dim = int(env.action_dim)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("RoboCasa env must expose integer action_dim.") from exc
    if action_dim != _ROBOCASA_LEROBOT_ACTION_DIM:
        raise ValueError(
            "RoboCasa action adapter supports the official 12-D LeRobot action "
            f"interface; env.action_dim={action_dim}."
        )
    return action_dim


def _read_native_action_bounds(env: Any) -> tuple[np.ndarray, np.ndarray] | None:
    action_spec = getattr(env, "action_spec", None)
    if action_spec is None:
        return None

    if callable(action_spec):
        try:
            action_spec = action_spec()
        except Exception as exc:
            raise RuntimeError("Failed to read RoboCasa action_spec.") from exc

    if not isinstance(action_spec, (tuple, list)) or len(action_spec) != 2:
        raise ValueError(
            "RoboCasa action_spec must be a (low, high) tuple/list when present."
        )

    action_dim = _native_action_dim(env)
    low = np.asarray(action_spec[0], dtype=np.float32)
    high = np.asarray(action_spec[1], dtype=np.float32)
    if low.shape != (action_dim,) or high.shape != (action_dim,):
        raise ValueError(
            "RoboCasa action_spec has unexpected shapes: "
            f"low={low.shape}, high={high.shape}, action_dim={action_dim}."
        )

    return low, high


def _build_action_space_from_env(env: Any) -> spaces.Box:
    # Keep this deliberately unbounded and task-invariant. RoboCasa native
    # action_spec bounds vary by task/controller details, but AsyncVectorEnv
    # requires every worker action_space to compare equal to the dummy parent
    # space. Bound enforcement happens in step() after mapping to native order.
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(_native_action_dim(env),),
        dtype=np.float32,
    )


def _clip_native_action_to_bounds(
    native_action: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, int]:
    if bounds is None:
        return native_action.astype(np.float32, copy=False), 0

    low, high = bounds
    below = native_action < low
    above = native_action > high
    clipped_count = int(below.sum() + above.sum())
    clipped = np.minimum(np.maximum(native_action, low), high).astype(
        np.float32, copy=False
    )
    return clipped, clipped_count


def _action_range_summary(values: np.ndarray) -> str:
    return f"min={float(np.min(values)):.4g} max={float(np.max(values)):.4g}"


class RobocasaEnv(gym.Env):
    """Gymnasium wrapper around a RoboCasa robosuite environment.

    Observation dict keys (compatible with lerobot's ``preprocess_observation``):
      - ``"pixels"``: ``{cam_name: (H, W, 3) uint8}`` for each camera
      - ``"robot_state"``: dict of float32 arrays keyed by the fixed official
        RoboCasa365 state ports

    ``task_description`` is a property (str), not an obs key.

    Args:
        task_name: Leaf task name, e.g. ``"CloseDrawer"``.
        image_size: Camera image height and width (square).
        seed: Random seed for robocasa scene initialization.
        camera_names: Cameras to include in observations.
        max_episode_steps: Episode truncation length.
        enable_render: Set False in the parent/dummy process to avoid
            initializing an OpenGL context before forking workers.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 20}

    # Robocasa camera images are stored upside-down; flip axis-0 (height) to correct.
    _FLIP_AXIS = 0
    _RESET_MAX_ATTEMPTS = 8
    _REBUILD_MAX_ATTEMPTS = 8

    _logger = _logger

    def __init__(
        self,
        task_name: str,
        split: Literal["all", "pretrain", "target"] = "all",
        image_size: int = 128,
        seed: int = 0,
        camera_names: list[str] | None = None,
        max_episode_steps: int = 500,
        enable_render: bool = True,
    ):
        super().__init__()
        if camera_names is None:
            camera_names = [
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ]

        self.task_name = task_name
        normalized_split = str(split).strip().lower()
        if normalized_split not in _VALID_SCENE_SPLITS:
            raise ValueError(
                f"Invalid RoboCasa split {split!r}; expected one of "
                f"{sorted(_VALID_SCENE_SPLITS)}."
            )
        self.split = cast(Literal["all", "pretrain", "target"], normalized_split)
        self.state_ports = list(ROBOCASA_STATE_PORTS)
        self.camera_names = list(camera_names)
        self.image_size = image_size
        self._max_episode_steps = int(max_episode_steps)
        self.enable_render = enable_render
        self._seed = int(seed)
        self._step_count = 0
        self._step_start_monotonic = time.monotonic()
        self._rebuild_count = 0
        self._native_action_clip_warnings_left = _MAX_NATIVE_ACTION_CLIP_WARNINGS
        self.env = self._create_env(seed=self._seed)
        env = self.env
        env.reset()  # required: populates layout_id etc. before get_ep_meta()
        self._task_description: str = env.get_ep_meta().get("lang", task_name)
        self._done = False

        # Observation space: pixels dict + robot_state dict (mirrors LIBERO).
        # Uses actual env obs to get correct state shapes.
        raw_obs = env._get_observations()
        self.observation_space = self._build_observation_space(raw_obs)
        self._native_action_bounds = _read_native_action_bounds(env)
        self.action_space = _build_action_space_from_env(env)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def task_description(self) -> str:
        """Natural-language task description (for ``add_envs_task``)."""
        return self._task_description

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reset_options = {} if options is None else dict(options)
        reseed_inner_env = bool(reset_options.pop("reseed_inner_env", True))
        current_seed = int(
            reset_options.pop(
                "episode_seed",
                self._seed if seed is None else int(seed),
            )
        )
        for attempt in range(self._RESET_MAX_ATTEMPTS):
            try:
                if reseed_inner_env:
                    self._prepare_inner_env_for_reset(seed=current_seed)
                raw_obs = self.env.reset()
                self._done = False
                self._seed = current_seed
                self._step_count = 0
                self._step_start_monotonic = time.monotonic()
                self._task_description = self.env.get_ep_meta().get(
                    "lang", self.task_name
                )
                return self._extract_obs(raw_obs), {}
            except Exception as exc:
                if not self._is_retryable_layout_error(exc):
                    raise
                if (attempt + 1) >= self._RESET_MAX_ATTEMPTS:
                    raise
                next_seed = current_seed + 1
                self._logger.warning(
                    "Robocasa reset failed with retryable layout error; "
                    "task=%s seed=%d attempt=%d/%d. Rebuilding env with seed=%d.",
                    self.task_name,
                    current_seed,
                    attempt + 1,
                    self._RESET_MAX_ATTEMPTS,
                    next_seed,
                )
                current_seed = self._rebuild_env(seed=next_seed)

        raise RuntimeError(
            "Unreachable: reset retry loop exhausted without return/raise."
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        native_action = self._prepare_native_action(action)
        raw_obs, _reward, terminated, info = self.env.step(native_action)
        is_success = bool(self.env._check_success())
        reward = 1.0 if is_success else 0.0
        info = dict(info)
        info["is_success"] = is_success
        self._done = (
            self._done
            or terminated
            or reward >= 1.0
            or self.env.timestep >= self._max_episode_steps
        )
        if self._done:
            info["final_info"] = {
                "task": self.task_name,
                "done": bool(terminated),
                "is_success": bool(is_success),
            }
        return self._extract_obs(raw_obs), reward, self._done, False, info

    def _prepare_native_action(self, action: np.ndarray) -> np.ndarray:
        native_action = _lerobot_action_to_native_order(action)
        clipped_action, clipped_count = _clip_native_action_to_bounds(
            native_action,
            getattr(self, "_native_action_bounds", None),
        )
        self._warn_if_native_action_clipped(
            before_clip=native_action,
            clipped_action=clipped_action,
            clipped_count=clipped_count,
        )
        return clipped_action

    def _warn_if_native_action_clipped(
        self,
        *,
        before_clip: np.ndarray,
        clipped_action: np.ndarray,
        clipped_count: int,
    ) -> None:
        if clipped_count <= 0:
            return
        warnings_left = int(getattr(self, "_native_action_clip_warnings_left", 0))
        if warnings_left <= 0:
            return
        self._native_action_clip_warnings_left = warnings_left - 1
        self._logger.warning(
            "Clipped RoboCasa native action before env.step: clipped=%d shape=%s "
            "before=(%s) after=(%s).",
            int(clipped_count),
            tuple(clipped_action.shape),
            _action_range_summary(before_clip),
            _action_range_summary(clipped_action),
        )

    def render(self) -> np.ndarray:
        cam = self.camera_names[0]
        frame = np.flip(
            self.env.sim.render(
                height=self.image_size,
                width=self.image_size,
                camera_name=cam,
            ),
            axis=self._FLIP_AXIS,
        ).astype(np.uint8)
        return frame

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_observation_space(self, raw_obs: dict[str, Any]) -> spaces.Dict:
        pixel_spaces: dict[str, gym.Space[Any]] = {
            cam: spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size, 3),
                dtype=np.uint8,
            )
            for cam in self.camera_names
        }
        state_spaces: dict[str, gym.Space[Any]] = {
            state_key_for_port(port): spaces.Box(
                low=-np.inf, high=np.inf, shape=raw_obs[port].shape, dtype=np.float32
            )
            for port in self.state_ports
        }
        return spaces.Dict(
            {
                "pixels": spaces.Dict(pixel_spaces),
                "robot_state": spaces.Dict(state_spaces),
            }
        )

    def _is_retryable_layout_error(self, exc: Exception) -> bool:
        return isinstance(exc, ValueError) and any(
            frag in str(exc) for frag in _RETRYABLE_LAYOUT_ERROR_FRAGMENTS
        )

    def _prepare_inner_env_for_reset(self, *, seed: int) -> None:
        inner_env = self.env
        cast(Any, inner_env).seed = int(seed)
        cast(Any, inner_env).rng = np.random.default_rng(int(seed))
        unset_ep_meta = getattr(inner_env, "unset_ep_meta", None)
        if callable(unset_ep_meta):
            unset_ep_meta()

    def _create_env(self, *, seed: int):
        from robocasa.utils.env_utils import create_env  # lazy: avoid OpenGL at import

        _install_objaverse_visual_mesh_guard()
        create_env_kwargs: dict[str, Any] = {
            "env_name": self.task_name,
            "split": self.split,
            "robots": "PandaOmron",
            "camera_names": self.camera_names,
            "camera_widths": self.image_size,
            "camera_heights": self.image_size,
            "seed": int(seed),
        }
        signature = inspect.signature(create_env)
        if "render_onscreen" in signature.parameters:
            # RoboCasa365 derives renderer / camera-observation flags internally.
            # We always keep the actual eval env headless; render-enabled means
            # offscreen rendering rather than an on-screen window.
            create_env_kwargs["render_onscreen"] = False
        else:
            create_env_kwargs.update(
                has_onscreen_renderer=False,
                has_offscreen_renderer=self.enable_render,
                use_camera_obs=self.enable_render,
            )
        env = create_env(**create_env_kwargs)
        return env

    def _rebuild_env(self, *, seed: int) -> int:
        last_retryable_exc: Exception | None = None
        for attempt in range(self._REBUILD_MAX_ATTEMPTS):
            candidate_seed = int(seed) + attempt
            try:
                new_env = self._create_env(seed=candidate_seed)
            except Exception as exc:
                if self._is_retryable_layout_error(exc):
                    if (attempt + 1) >= self._REBUILD_MAX_ATTEMPTS:
                        last_retryable_exc = exc
                        break
                    self._logger.warning(
                        "Robocasa _create_env failed during rebuild with retryable layout error; "
                        "task=%s seed=%d attempt=%d/%d. Retrying with seed=%d.",
                        self.task_name,
                        candidate_seed,
                        attempt + 1,
                        self._REBUILD_MAX_ATTEMPTS,
                        candidate_seed + 1,
                    )
                    last_retryable_exc = exc
                    continue
                raise

            try:
                native_action_bounds = _read_native_action_bounds(new_env)
            except Exception:
                try:
                    new_env.close()
                finally:
                    raise

            old_env = self.env
            self._native_action_bounds = native_action_bounds
            self.env = new_env
            try:
                old_env.close()
            except Exception:
                self._logger.warning(
                    "Robocasa old env close failed during rebuild; task=%s seed=%d",
                    self.task_name,
                    candidate_seed,
                    exc_info=True,
                )
            self._rebuild_count += 1
            return candidate_seed

        assert last_retryable_exc is not None
        raise RuntimeError(
            f"Failed to rebuild RobocasaEnv for task={self.task_name!r} "
            f"after {self._REBUILD_MAX_ATTEMPTS} attempts from seed={int(seed)}."
        ) from last_retryable_exc

    def _extract_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        pixels: dict[str, np.ndarray] = {}
        if self.enable_render:
            for cam in self.camera_names:
                # robosuite stores images upside-down; flip height axis.
                pixels[cam] = np.flip(
                    raw_obs[f"{cam}_image"], axis=self._FLIP_AXIS
                ).astype(np.uint8)
        else:
            for cam in self.camera_names:
                pixels[cam] = np.zeros(
                    (self.image_size, self.image_size, 3), dtype=np.uint8
                )

        robot_state = {
            state_key_for_port(port): raw_obs[port].astype(np.float32)
            for port in self.state_ports
        }

        return {"pixels": pixels, "robot_state": robot_state}
