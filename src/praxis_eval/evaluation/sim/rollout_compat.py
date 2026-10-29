# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LeRobot rollout compatibility helpers for vectorized env pools."""

from __future__ import annotations

import logging
import multiprocessing
import queue
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import gymnasium as gym
import numpy as np
import torch
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video
from lerobot.utils.utils import inside_slurm
from torch import nn
from tqdm import trange

from praxis_eval.evaluation.sim.metrics import TaskEvalResult

_logger = logging.getLogger(__name__)


class _VectorEnvWithCall(Protocol):
    num_envs: int

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any: ...


class _EnvAttrStub:
    """Lightweight env placeholder for LeRobot env attribute checks."""

    def __init__(self) -> None:
        self.task_description = ""
        self.task = ""


class _RolloutEnvCompat:
    """Adapter exposing LeRobot-expected attrs across vector env variants."""

    def __init__(self, env: gym.vector.VectorEnv) -> None:
        self._env = env
        self.num_envs = int(env.num_envs)
        self._action_sanitizer = _RolloutActionSanitizer()

        if isinstance(env, gym.vector.AsyncVectorEnv):
            self._has_task_description = _async_env_has_string_attr(
                env, "task_description"
            )
            self._has_task = _async_env_has_string_attr(env, "task")
        else:
            self._has_task_description = _vector_env_has_string_attr(
                env, "task_description"
            )
            self._has_task = _vector_env_has_string_attr(env, "task")

        # LeRobot's checks require both attrs to exist on `env.envs[0]`.
        self.envs: list[_EnvAttrStub] = []
        for _ in range(self.num_envs):
            self.envs.append(_EnvAttrStub())

    def _call_or_empty(self, attr_name: str) -> list[str]:
        env = cast(_VectorEnvWithCall, self._env)
        try:
            values = env.call(attr_name)
        except Exception:
            return [""] * self.num_envs
        normalized = _string_list_or_none(values)
        if normalized is None:
            return [""] * self.num_envs
        return normalized

    def call(self, name: str, *args, **kwargs):
        if name == "task_description" and not self._has_task_description:
            if self._has_task:
                return self._call_or_empty("task")
            return [""] * self.num_envs
        if name == "task" and not self._has_task:
            if self._has_task_description:
                return self._call_or_empty("task_description")
            return [""] * self.num_envs
        env = cast(_VectorEnvWithCall, self._env)
        return env.call(name, *args, **kwargs)

    def step(self, action):
        action = self._action_sanitizer.sanitize(
            action,
            action_space=getattr(self._env, "action_space", None),
        )
        try:
            return self._env.step(action)
        except EOFError:
            base_env = _unwrap_rollout_env(cast(gym.vector.VectorEnv, self._env))
            if isinstance(base_env, gym.vector.AsyncVectorEnv):
                _log_async_vector_env_eof(
                    base_env,
                    timeout_sec=None,
                    action=action,
                )
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _LaneHorizonVectorEnv:
    """Ends each lane's episode at that lane's own ``_max_episode_steps``.

    One pool can mix tasks with different horizons, e.g. ``libero_spatial`` (280
    steps) and ``libero_10`` (520), and not every env ends at its horizon on its
    own: LeRobot's ``LiberoEnv`` never truncates. The rollout loop reads lane 0's
    ``_max_episode_steps`` as the step cap for the whole wave, so this adapter
    reports the longest horizon there and marks a lane ``truncated`` once it has
    taken its own horizon in steps since the last reset. ``_StepTimeoutVectorEnv``
    drives async pools through ``reset_wait`` and ``step_wait``, so those do the
    same bookkeeping as ``reset`` and ``step``.
    """

    def __init__(self, env: _RolloutEnvCompat) -> None:
        self._env = env
        self.num_envs = int(env.num_envs)
        self._horizons: np.ndarray | None = None
        self._elapsed = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, **kwargs):
        return self._start_episodes(self._env.reset(**kwargs))

    def reset_wait(self, **kwargs):
        return self._start_episodes(self._env.reset_wait(**kwargs))

    def step(self, action):
        return self._end_lanes_at_horizon(self._env.step(action))

    def step_wait(self, **kwargs):
        return self._end_lanes_at_horizon(self._env.step_wait(**kwargs))

    def call(self, name: str, *args, **kwargs):
        if name == "_max_episode_steps":
            return [int(self._lane_horizons().max())] * self.num_envs
        return self._env.call(name, *args, **kwargs)

    def _start_episodes(self, reset_result):
        # Read after the reset, like LeRobot's rollout: a lane's horizon follows
        # the task its job prepared.
        values = self._env.call("_max_episode_steps")
        horizons = _int_list_or_none(values)
        if horizons is None or len(horizons) != self.num_envs:
            raise ValueError(
                f"Expected one integer _max_episode_steps per lane, got {values!r}."
            )
        self._horizons = np.asarray(horizons, dtype=np.int64)
        self._elapsed = np.zeros(self.num_envs, dtype=np.int64)
        return reset_result

    def _end_lanes_at_horizon(self, step_result):
        horizons = self._lane_horizons()
        observation, reward, terminated, truncated, info = step_result
        self._elapsed += 1
        truncated = np.asarray(truncated, dtype=bool) | (self._elapsed >= horizons)
        return observation, reward, terminated, truncated, info

    def _lane_horizons(self) -> np.ndarray:
        if self._horizons is None:
            raise RuntimeError("Call reset() before step() or reading lane horizons.")
        return self._horizons

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _StepTimeoutVectorEnv:
    """Vector env adapter that enforces reset/step timeouts on async pools."""

    def __init__(
        self,
        env: gym.vector.VectorEnv | _RolloutEnvCompat | _LaneHorizonVectorEnv,
        *,
        timeout_sec: float,
        phase_heartbeat: Callable[[str], None] | None = None,
    ) -> None:
        self._env = env
        self._timeout_sec = float(timeout_sec)
        self._phase_heartbeat = phase_heartbeat
        self.num_envs = int(env.num_envs)
        self.envs = getattr(env, "envs", [])
        self._action_sanitizer = _RolloutActionSanitizer()

    def reset(self, **kwargs):
        base_env = _unwrap_rollout_env(cast(gym.vector.VectorEnv, self._env))
        if isinstance(base_env, gym.vector.AsyncVectorEnv):
            self._env.reset_async(**kwargs)
            try:
                result = self._env.reset_wait(timeout=self._timeout_sec)
                if self._phase_heartbeat is not None:
                    self._phase_heartbeat("reset_wait_return")
                return result
            except multiprocessing.TimeoutError as exc:
                raise RuntimeError(
                    f"Timed out waiting for async env reset after {self._timeout_sec:.1f}s."
                ) from exc
        return self._env.reset(**kwargs)

    def step(self, action):
        base_env = _unwrap_rollout_env(cast(gym.vector.VectorEnv, self._env))
        if isinstance(base_env, gym.vector.AsyncVectorEnv):
            action = self._action_sanitizer.sanitize(
                action,
                action_space=getattr(base_env, "action_space", None),
            )
            self._env.step_async(action)
            try:
                result = self._env.step_wait(timeout=self._timeout_sec)
                if self._phase_heartbeat is not None:
                    self._phase_heartbeat("step_wait_return")
                return result
            except EOFError:
                _log_async_vector_env_eof(
                    base_env,
                    timeout_sec=self._timeout_sec,
                    action=action,
                )
                raise
            except multiprocessing.TimeoutError as exc:
                raise RuntimeError(
                    f"Timed out waiting for async env step after {self._timeout_sec:.1f}s."
                ) from exc
        return self._env.step(action)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _RolloutActionSanitizer:
    """Sanitize policy actions before they cross async env worker boundaries."""

    def __init__(self, *, max_warnings: int = 5) -> None:
        self._warnings_left = int(max_warnings)

    def sanitize(self, action: Any, *, action_space: Any) -> Any:
        action_np = np.asarray(action)
        if action_np.dtype.kind not in {"f", "i", "u"}:
            return action

        target_dtype = _action_dtype(action_np, action_space)
        working = action_np.astype(target_dtype, copy=False)
        nonfinite_count = int((~np.isfinite(working)).sum())

        bounds = _box_bounds_for_action_shape(working, action_space)
        out_of_bounds_count = 0
        if bounds is not None:
            low, high = bounds
            finite_low = np.isfinite(low)
            finite_high = np.isfinite(high)
            finite_action = np.isfinite(working)
            below = finite_action & finite_low & (working < low)
            above = finite_action & finite_high & (working > high)
            out_of_bounds_count = int(below.sum() + above.sum())

        if nonfinite_count == 0 and out_of_bounds_count == 0:
            return working

        sanitized = np.array(working, copy=True)
        if nonfinite_count > 0:
            sanitized = np.nan_to_num(sanitized, nan=0.0, posinf=0.0, neginf=0.0)

        if bounds is not None:
            low, high = bounds
            clip_low = np.where(np.isfinite(low), low, sanitized)
            clip_high = np.where(np.isfinite(high), high, sanitized)
            sanitized = np.minimum(np.maximum(sanitized, clip_low), clip_high)

        if self._warnings_left > 0:
            self._warnings_left -= 1
            _logger.warning(
                "Sanitized rollout action before env.step: nonfinite=%d out_of_bounds=%d "
                "shape=%s before=(%s) after=(%s).",
                nonfinite_count,
                out_of_bounds_count,
                tuple(action_np.shape),
                _finite_range_summary(working),
                _finite_range_summary(sanitized),
            )

        return sanitized.astype(target_dtype, copy=False)


def _action_dtype(action_np: np.ndarray, action_space: Any) -> np.dtype:
    if isinstance(action_space, gym.spaces.Box):
        try:
            dtype = np.dtype(action_space.dtype)
            if dtype.kind in {"f", "i", "u"}:
                return dtype
        except TypeError:
            pass
    if action_np.dtype.kind == "f":
        return action_np.dtype
    return np.dtype(np.float32)


def _box_bounds_for_action_shape(
    action_np: np.ndarray,
    action_space: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(action_space, gym.spaces.Box):
        return None

    low = np.asarray(action_space.low, dtype=action_np.dtype)
    high = np.asarray(action_space.high, dtype=action_np.dtype)
    try:
        return (
            np.broadcast_to(low, action_np.shape),
            np.broadcast_to(high, action_np.shape),
        )
    except ValueError:
        _logger.debug(
            "Cannot broadcast action_space bounds shape=%s/%s to action shape=%s.",
            low.shape,
            high.shape,
            action_np.shape,
        )
        return None


def _finite_range_summary(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "all_nonfinite"
    return f"min={float(finite.min()):.4g} max={float(finite.max()):.4g}"


def _log_async_vector_env_eof(
    base_env: gym.vector.AsyncVectorEnv,
    *,
    timeout_sec: float | None,
    action: Any | None = None,
) -> None:
    timeout_label = "no timeout" if timeout_sec is None else f"timeout={timeout_sec}s"
    proc_entries: list[str] = []
    for idx, proc in enumerate(getattr(base_env, "processes", []) or []):
        pid = getattr(proc, "pid", None)
        is_alive_fn = getattr(proc, "is_alive", None)
        is_alive = is_alive_fn() if callable(is_alive_fn) else None
        exitcode = getattr(proc, "exitcode", None)
        proc_entries.append(
            f"idx={idx} pid={pid} is_alive={is_alive} exitcode={exitcode}"
        )
    if proc_entries:
        _logger.error(
            "EOFError in AsyncVectorEnv.step_wait(%s); worker statuses: %s",
            timeout_label,
            " | ".join(proc_entries),
        )
    else:
        _logger.error(
            "EOFError in AsyncVectorEnv.step_wait(%s); no worker process metadata found.",
            timeout_label,
        )

    _log_dead_worker_action_summaries(base_env, action=action)

    error_queue = getattr(base_env, "error_queue", None)
    if error_queue is not None:
        drained = 0
        while True:
            try:
                idx, exctype, value, trace = error_queue.get_nowait()
            except queue.Empty:
                break
            except Exception as queue_exc:
                _logger.error(
                    "Failed draining AsyncVectorEnv error_queue after EOFError: %r",
                    queue_exc,
                )
                break
            drained += 1
            _logger.error(
                "AsyncVectorEnv error_queue item idx=%s exctype=%s value=%r trace=%s",
                idx,
                getattr(exctype, "__name__", str(exctype)),
                value,
                trace,
            )
        if drained == 0:
            _logger.error(
                "AsyncVectorEnv error_queue had no pending items after EOFError."
            )

    stderr_like_attrs = [
        name
        for name in (
            "stderr_pipe",
            "stderr_pipes",
            "_stderr_pipe",
            "_stderr_pipes",
        )
        if hasattr(base_env, name)
    ]
    if stderr_like_attrs:
        _logger.error(
            "AsyncVectorEnv has stderr-like attrs after EOFError: %s",
            ", ".join(stderr_like_attrs),
        )
    else:
        _logger.error(
            "AsyncVectorEnv exposes no stderr pipe attrs; worker stderr recovery unavailable."
        )


def _log_dead_worker_action_summaries(
    base_env: gym.vector.AsyncVectorEnv,
    *,
    action: Any | None,
) -> None:
    if action is None:
        return
    action_np = np.asarray(action)
    if action_np.ndim < 2:
        return

    for idx, proc in enumerate(getattr(base_env, "processes", []) or []):
        exitcode = getattr(proc, "exitcode", None)
        if exitcode is None or idx >= action_np.shape[0]:
            continue

        lane_action = action_np[idx]
        nonfinite = int((~np.isfinite(lane_action)).sum())
        _logger.error(
            "AsyncVectorEnv dead worker action summary idx=%d exitcode=%s "
            "nonfinite=%d shape=%s range=(%s) values=%s",
            idx,
            exitcode,
            nonfinite,
            tuple(lane_action.shape),
            _finite_range_summary(lane_action),
            np.array2string(
                lane_action,
                precision=5,
                separator=", ",
                threshold=64,
            ),
        )


def _vector_env_has_string_attr(env: gym.vector.VectorEnv, attr_name: str) -> bool:
    """Return True if ``env.call(attr_name)`` yields a list[str]-like payload."""
    callable_env = cast(_VectorEnvWithCall, env)
    try:
        values = callable_env.call(attr_name)
    except Exception:
        return False

    return _string_list_or_none(values) is not None


def _async_env_has_string_attr(env: gym.vector.AsyncVectorEnv, attr_name: str) -> bool:
    """Return True if ``env.call(attr_name)`` yields a list[str]-like payload."""
    callable_env = cast(_VectorEnvWithCall, env)
    try:
        exists = callable_env.call("has_wrapper_attr", attr_name)
    except Exception:
        return False

    exists_list = _list_or_none(exists)
    if exists_list is None or not all(bool(v) for v in exists_list):
        return False

    try:
        values = callable_env.call(attr_name)
    except Exception:
        return False

    return _string_list_or_none(values) is not None


def _list_or_none(values: Any) -> list[Any] | None:
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list) or len(values) == 0:
        return None
    return values


def _string_list_or_none(values: Any) -> list[str] | None:
    value_list = _list_or_none(values)
    if value_list is None or not all(isinstance(v, str) for v in value_list):
        return None
    return value_list


def _int_list_or_none(values: Any) -> list[int] | None:
    value_list = _list_or_none(values)
    if value_list is None:
        return None
    normalized: list[int] = []
    for value in value_list:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            return None
    return normalized


def _unwrap_rollout_env(env: gym.vector.VectorEnv) -> gym.vector.VectorEnv:
    """Return the vector env under this module's rollout adapters."""
    while isinstance(
        env, _RolloutEnvCompat | _LaneHorizonVectorEnv | _StepTimeoutVectorEnv
    ):
        env = env._env
    return env


def _resolve_render_fps(env: gym.vector.VectorEnv) -> int:
    """Resolve render FPS from env metadata with a safe fallback."""
    base_env = _unwrap_rollout_env(env)
    metadata = getattr(getattr(base_env, "unwrapped", base_env), "metadata", None)
    if isinstance(metadata, dict) and "render_fps" in metadata:
        try:
            return int(metadata["render_fps"])
        except Exception:
            pass
    return 30


def _extract_task_descriptions(env: gym.vector.VectorEnv) -> list[str]:
    """Best-effort extraction of per-lane task descriptions from vector env."""
    callable_env = cast(_VectorEnvWithCall, env)
    for attr in ("task_description", "task"):
        try:
            values = callable_env.call(attr)
        except Exception:
            continue
        normalized = _string_list_or_none(values)
        if normalized is not None:
            return normalized
    return [""] * int(getattr(env, "num_envs", 0))


def process_observation(
    env: Any,
    observation: dict[str, Any],
    env_preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Turn one raw vector-env observation batch into the batch policies see.

    Applies LeRobot's ``preprocess_observation``, adds the per-lane task strings
    with ``add_envs_task``, and runs the env preprocessor. ``env`` must expose
    LeRobot's task attributes, as ``_RolloutEnvCompat`` does. The eval rollout
    and ``EvalVectorEnv`` both call this function, so they cannot drift apart.
    """
    observation = preprocess_observation(observation)
    observation = add_envs_task(env, observation)
    return env_preprocessor(observation)


def postprocess_action(
    action: torch.Tensor,
    env_postprocessor: Callable[[dict[str, Any]], dict[str, Any]],
) -> np.ndarray:
    """Run the env postprocessor on a batched action and return a numpy array."""
    action_numpy = env_postprocessor({ACTION: action})[ACTION].to("cpu").numpy()
    if action_numpy.ndim != 2:
        raise ValueError(
            f"Action dimensions should be (batch, action_dim), got {action_numpy.shape}."
        )
    return action_numpy


def step_successes(info: dict[str, Any], num_envs: int) -> np.ndarray:
    """Return per-lane success flags for one vector-env step.

    Success is read from ``info["final_info"]["is_success"]``, the same field
    LeRobot's rollout reads, so lanes that did not finish report ``False``.
    """
    if "final_info" not in info:
        return np.zeros(num_envs, dtype=bool)
    final_info = info["final_info"]
    if not isinstance(final_info, dict):
        raise RuntimeError(
            "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0)."
        )
    return np.asarray(final_info["is_success"], dtype=bool)


def _rollout(
    *,
    env: Any,
    policy: nn.Module,
    env_preprocessor: Callable[[Any], Any],
    env_postprocessor: Callable[[Any], Any],
    preprocessor: Callable[[Any], Any],
    postprocessor: Callable[[Any], Any],
    seeds: list[int] | None,
    render_callback: Callable[[Any], None] | None,
) -> dict[str, torch.Tensor]:
    """Run one batched episode per lane until every lane is done.

    Follows LeRobot 0.4.4's ``rollout``: lanes keep stepping after they finish,
    ``done`` is cumulative, and every lane is done after ``_max_episode_steps``
    steps. Returns ``(num_envs, steps)`` tensors for ``reward``, ``success``, and
    ``done``.
    """
    policy.reset()
    observation, _ = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    num_envs = int(env.num_envs)
    max_steps = int(env.call("_max_episode_steps")[0])
    done = np.zeros(num_envs, dtype=bool)
    rewards: list[torch.Tensor] = []
    successes: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    with trange(
        max_steps, desc="Rollout", disable=inside_slurm(), leave=False
    ) as progbar:
        step = 0
        while not np.all(done) and step < max_steps:
            batch = preprocessor(
                process_observation(env, observation, env_preprocessor)
            )
            with torch.inference_mode():
                action = policy.select_action(batch)
            action_numpy = postprocess_action(postprocessor(action), env_postprocessor)

            observation, reward, terminated, truncated, info = env.step(action_numpy)
            if render_callback is not None:
                render_callback(env)

            done = terminated | truncated | done
            if step + 1 == max_steps:
                done = np.ones_like(done, dtype=bool)
            rewards.append(torch.from_numpy(np.asarray(reward)))
            successes.append(torch.from_numpy(step_successes(info, num_envs)))
            dones.append(torch.from_numpy(np.asarray(done, dtype=bool)))

            step += 1
            succ_rate = torch.stack(successes, dim=1).any(dim=1).float().mean()
            progbar.set_postfix({"succ_rate": f"{succ_rate.item() * 100:.1f}%"})
            progbar.update()

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return {
        "reward": torch.stack(rewards, dim=1),
        "success": torch.stack(successes, dim=1),
        "done": torch.stack(dones, dim=1),
    }


def _terminal_rollout_metrics(
    rollout_data: dict[str, torch.Tensor],
) -> tuple[list[float], list[float], list[bool], list[int], torch.Tensor]:
    """Aggregate rollout metrics over timesteps through first terminal step."""
    done = rollout_data["done"].bool()
    n_steps = int(done.shape[1])
    first_done = torch.argmax(done.to(torch.int64), dim=1)
    last_step = torch.full_like(first_done, max(n_steps - 1, 0))
    done_indices = torch.where(done.any(dim=1), first_done, last_step)
    valid = torch.arange(n_steps, device=done.device).unsqueeze(
        0
    ) <= done_indices.unsqueeze(1)

    rewards = rollout_data["reward"].float()
    sum_rewards = rewards.masked_fill(~valid, 0).sum(dim=1).tolist()
    max_rewards = rewards.masked_fill(~valid, -torch.inf).amax(dim=1).tolist()
    successes = (rollout_data["success"].bool() & valid).any(dim=1).tolist()
    lengths = (done_indices + 1).tolist()
    return sum_rewards, max_rewards, successes, lengths, done_indices


def evaluate_policy_on_pooled_env(
    *,
    env: gym.vector.VectorEnv,
    policy: nn.Module,
    seeds: list[int] | None = None,
    preprocessor=lambda x: x,
    postprocessor=lambda x: x,
    env_preprocessor=lambda x: x,
    env_postprocessor=lambda x: x,
    step_timeout_sec: float | int | None = None,
    max_episodes_rendered_by_env: list[int] | None = None,
    videos_dirs_by_env: list[Path | None] | None = None,
    video_start_index_by_env: list[int] | None = None,
    phase_heartbeat: Callable[[str], None] | None = None,
) -> list[TaskEvalResult]:
    """Run one rollout on a persistent pooled vec env and return per-lane stats."""
    num_envs = int(env.num_envs)
    if num_envs < 1:
        return []

    max_episodes_rendered_by_env = (
        [0] * num_envs
        if max_episodes_rendered_by_env is None
        else list(max_episodes_rendered_by_env)
    )
    if len(max_episodes_rendered_by_env) != num_envs:
        raise ValueError(
            f"max_episodes_rendered_by_env length mismatch: expected {num_envs}, got {len(max_episodes_rendered_by_env)}."
        )

    videos_dirs_by_env_list: list[Path | None] = (
        [None for _ in range(num_envs)]
        if videos_dirs_by_env is None
        else list(videos_dirs_by_env)
    )
    if len(videos_dirs_by_env_list) != num_envs:
        raise ValueError(
            f"videos_dirs_by_env length mismatch: expected {num_envs}, got {len(videos_dirs_by_env_list)}."
        )

    video_start_index_by_env = (
        [0] * num_envs
        if video_start_index_by_env is None
        else [int(v) for v in video_start_index_by_env]
    )
    if len(video_start_index_by_env) != num_envs:
        raise ValueError(
            f"video_start_index_by_env length mismatch: expected {num_envs}, got {len(video_start_index_by_env)}."
        )

    for lane_idx, max_rendered in enumerate(max_episodes_rendered_by_env):
        if int(max_rendered) > 0 and videos_dirs_by_env_list[lane_idx] is None:
            raise ValueError(
                f"videos_dirs_by_env[{lane_idx}] is required when max_episodes_rendered_by_env[{lane_idx}] > 0."
            )

    lane_frames: list[list[np.ndarray]] = [[] for _ in range(num_envs)]

    def _render_frame(vec_env: gym.vector.VectorEnv) -> None:
        base_env = _unwrap_rollout_env(vec_env)
        rendered: list[Any]
        if isinstance(base_env, gym.vector.SyncVectorEnv):
            rendered = [base_env.envs[i].render() for i in range(base_env.num_envs)]
        elif isinstance(base_env, gym.vector.AsyncVectorEnv):
            frames = base_env.call("render")
            if isinstance(frames, tuple):
                frames = list(frames)
            rendered = list(frames) if isinstance(frames, list) else []
        else:
            frame = vec_env.render()
            if isinstance(frame, list):
                rendered = frame
            elif isinstance(frame, tuple):
                rendered = list(frame)
            elif isinstance(frame, np.ndarray) and frame.ndim >= 4:
                rendered = [frame[i] for i in range(min(frame.shape[0], num_envs))]
            else:
                rendered = [frame]

        for lane_idx, frame in enumerate(rendered):
            if lane_idx >= num_envs:
                break
            if frame is None:
                continue
            if int(max_episodes_rendered_by_env[lane_idx]) <= 0:
                continue
            lane_frames[lane_idx].append(frame)

    rollout_env: _LaneHorizonVectorEnv | _StepTimeoutVectorEnv = _LaneHorizonVectorEnv(
        _RolloutEnvCompat(env)
    )
    if step_timeout_sec is not None and float(step_timeout_sec) > 0.0:
        rollout_env = _StepTimeoutVectorEnv(
            rollout_env,
            timeout_sec=float(step_timeout_sec),
            phase_heartbeat=phase_heartbeat,
        )
    rollout_data = _rollout(
        env=rollout_env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        seeds=seeds,
        render_callback=_render_frame
        if any(int(x) > 0 for x in max_episodes_rendered_by_env)
        else None,
    )

    sum_rewards, max_rewards, successes, lengths, done_indices = (
        _terminal_rollout_metrics(rollout_data)
    )
    lane_task_descs = _extract_task_descriptions(env)

    results: list[TaskEvalResult] = []
    for lane_idx in range(num_envs):
        video_paths: list[str] = []
        max_rendered = int(max_episodes_rendered_by_env[lane_idx])
        if max_rendered > 0:
            videos_dir = videos_dirs_by_env_list[lane_idx]
            if videos_dir is not None and len(lane_frames[lane_idx]) > 0:
                videos_dir.mkdir(parents=True, exist_ok=True)
                done_index = int(done_indices[lane_idx].item())
                video_start_index = int(video_start_index_by_env[lane_idx])
                video_path = videos_dir / f"eval_episode_{video_start_index}.mp4"
                fps = _resolve_render_fps(env)
                stacked = np.stack(lane_frames[lane_idx], axis=0)
                video_paths.append(str(video_path))
                write_video(str(video_path), stacked[: done_index + 1], fps)

        task_desc = lane_task_descs[lane_idx] if lane_idx < len(lane_task_descs) else ""
        results.append(
            TaskEvalResult(
                task_group="",
                task_id=0,
                task_description=task_desc,
                sum_rewards=[float(sum_rewards[lane_idx])],
                max_rewards=[float(max_rewards[lane_idx])],
                successes=[bool(successes[lane_idx])],
                lengths=[int(lengths[lane_idx])],
                video_paths=video_paths,
            )
        )

    return results
