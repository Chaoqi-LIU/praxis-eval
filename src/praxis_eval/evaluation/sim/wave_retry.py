# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Retry execution for one prepared pooled-rollout wave."""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from praxis_eval.envs.eval_pool import EvalLaneJob, EvalPoolHandle
from praxis_eval.evaluation.sim.metrics import TaskEvalResult
from praxis_eval.evaluation.sim.rollout_compat import evaluate_policy_on_pooled_env

DEFAULT_ROLLOUT_FAILURE_RETRIES = 1
_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutFailureRetryPolicy:
    """Bounded retry policy for native async rollout failures."""

    max_retries: int = DEFAULT_ROLLOUT_FAILURE_RETRIES

    def __post_init__(self) -> None:
        max_retries = int(self.max_retries)
        if max_retries < 0:
            raise ValueError(
                f"rollout_failure_retries must be >= 0, got {self.max_retries}."
            )
        object.__setattr__(self, "max_retries", max_retries)

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1

    def should_retry(self, *, attempt_idx: int, exc: BaseException) -> bool:
        return int(
            attempt_idx
        ) < self.max_attempts - 1 and is_retryable_rollout_failure(exc)


@dataclass(frozen=True)
class RolloutRenderPlan:
    """Per-lane rendering controls for a rollout wave."""

    max_episodes_rendered_by_env: list[int]
    videos_dirs_by_env: list[Path | None]
    video_start_index_by_env: list[int]


class PooledRolloutWaveRunner:
    """Runs prepared rollout waves with bounded pool rebuild/retry semantics."""

    def __init__(
        self,
        *,
        policy: nn.Module,
        retry_policy: RolloutFailureRetryPolicy,
        preprocessor: Callable[[Any], Any],
        postprocessor: Callable[[Any], Any],
        env_preprocessor: Callable[[Any], Any],
        env_postprocessor: Callable[[Any], Any],
        rollout_step_timeout_sec: float | int | None,
        phase_heartbeat: Callable[[str], None] | None,
    ) -> None:
        self._policy = policy
        self._retry_policy = retry_policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._env_preprocessor = env_preprocessor
        self._env_postprocessor = env_postprocessor
        self._rollout_step_timeout_sec = rollout_step_timeout_sec
        self._phase_heartbeat = phase_heartbeat

    def run(
        self,
        *,
        eval_pool: EvalPoolHandle,
        prepared_jobs: list[EvalLaneJob | None],
        num_parallel_env: int,
        wave_idx: int,
        total_waves: int,
        seeds: list[int] | None,
        render_plan: RolloutRenderPlan,
    ) -> list[TaskEvalResult]:
        for attempt_idx in range(self._retry_policy.max_attempts):
            if attempt_idx > 0:
                self._mark_retry_phase(
                    "rollout_retry_prepare_begin",
                    wave_idx=wave_idx,
                    attempt_idx=attempt_idx,
                )
                prepare_rebuilt_eval_pool(
                    eval_pool=eval_pool,
                    prepared_jobs=prepared_jobs,
                    num_parallel_env=num_parallel_env,
                )
                self._mark_retry_phase(
                    "rollout_retry_prepare_end",
                    wave_idx=wave_idx,
                    attempt_idx=attempt_idx,
                )

            if eval_pool.env_pool is None:
                raise RuntimeError("Eval pool is missing before rollout.")

            try:
                return evaluate_policy_on_pooled_env(
                    env=eval_pool.env_pool,
                    policy=self._policy,
                    seeds=seeds,
                    preprocessor=self._preprocessor,
                    postprocessor=self._postprocessor,
                    env_preprocessor=self._env_preprocessor,
                    env_postprocessor=self._env_postprocessor,
                    step_timeout_sec=self._rollout_step_timeout_sec,
                    max_episodes_rendered_by_env=render_plan.max_episodes_rendered_by_env,
                    videos_dirs_by_env=render_plan.videos_dirs_by_env,
                    video_start_index_by_env=render_plan.video_start_index_by_env,
                    phase_heartbeat=self._phase_heartbeat,
                )
            except Exception as exc:
                if self._retry_policy.should_retry(
                    attempt_idx=attempt_idx,
                    exc=exc,
                ):
                    _log_rollout_failure(
                        prepared_jobs=prepared_jobs,
                        wave_idx=wave_idx,
                        total_waves=total_waves,
                        attempt_idx=attempt_idx,
                        max_attempts=self._retry_policy.max_attempts,
                        retrying=True,
                    )
                    eval_pool.close(terminate=True)
                    continue
                _log_rollout_failure(
                    prepared_jobs=prepared_jobs,
                    wave_idx=wave_idx,
                    total_waves=total_waves,
                    attempt_idx=attempt_idx,
                    max_attempts=self._retry_policy.max_attempts,
                    retrying=False,
                )
                raise

        raise RuntimeError("Unreachable: rollout retry loop exhausted without return.")

    def _mark_retry_phase(
        self,
        label: str,
        *,
        wave_idx: int,
        attempt_idx: int,
    ) -> None:
        if self._phase_heartbeat is None:
            return
        self._phase_heartbeat(f"{label} wave={wave_idx} attempt={attempt_idx + 1}")


def lane_context(prepared_jobs: list[EvalLaneJob | None]) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for lane_idx, job in enumerate(prepared_jobs):
        entry: dict[str, Any] = {"lane_idx": int(lane_idx)}
        if job is None:
            entry["job"] = None
        else:
            entry.update(
                task_group=str(job.task_group),
                task_id=int(job.task_id),
                eval_idx=int(job.eval_idx),
                episode_index=int(job.episode_index),
            )
        context.append(entry)
    return context


def prepare_rebuilt_eval_pool(
    *,
    eval_pool: EvalPoolHandle,
    prepared_jobs: list[EvalLaneJob | None],
    num_parallel_env: int,
) -> None:
    eval_pool.prepare_jobs(prepared_jobs)
    if eval_pool.env_pool is None:
        raise RuntimeError(
            "Eval pool retry prepare_jobs completed without initializing env_pool."
        )
    prepared_num_envs = int(eval_pool.env_pool.num_envs)
    if prepared_num_envs != int(num_parallel_env):
        raise RuntimeError(
            "Eval pool lane count mismatch after retry prepare_jobs: "
            f"handle has {num_parallel_env}, env_pool has {prepared_num_envs}; "
            f"lane_context={lane_context(prepared_jobs)}"
        )


def is_retryable_rollout_failure(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(
            current,
            (
                EOFError,
                BrokenPipeError,
                ConnectionResetError,
                multiprocessing.TimeoutError,
            ),
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _log_rollout_failure(
    *,
    prepared_jobs: list[EvalLaneJob | None],
    wave_idx: int,
    total_waves: int,
    attempt_idx: int,
    max_attempts: int,
    retrying: bool,
) -> None:
    retry_msg = (
        "rebuilding env pool and retrying same lane assignments"
        if retrying
        else "no retries left or failure is not retryable"
    )
    _logger.exception(
        "Pooled sim rollout failed in wave %d/%d on attempt %d/%d; %s; lane_context=%s",
        wave_idx,
        total_waves,
        attempt_idx + 1,
        max_attempts,
        retry_msg,
        lane_context(prepared_jobs),
    )
