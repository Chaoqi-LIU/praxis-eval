# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Simulation evaluation runner on persistent async env pools."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch import nn
from tqdm import tqdm

from praxis_eval.contracts import ActionSpec
from praxis_eval.envs.eval_pool import EvalLaneJob, EvalPoolHandle
from praxis_eval.evaluation.config import require_positive_int
from praxis_eval.evaluation.sim.adapters import LocalPolicyAdapter
from praxis_eval.evaluation.sim.metrics import (
    TaskEvalResult,
    summarize_task_eval_results,
)
from praxis_eval.evaluation.sim.wave_retry import (
    PooledRolloutWaveRunner,
    RolloutFailureRetryPolicy,
    RolloutRenderPlan,
    lane_context,
)
from praxis_eval.types import Policy


def _identity(x):
    return x


def evaluate_policy_on_env_pool(
    *,
    tasks: list[tuple[str, int]],
    eval_pool: EvalPoolHandle,
    policy: Policy | nn.Module,
    num_eval_per_task: int,
    start_seed: int | None = None,
    device: str | Any = "cpu",
    preprocessor=_identity,
    postprocessor=_identity,
    env_preprocessor=_identity,
    env_postprocessor=_identity,
    policy_kwargs: dict[str, Any] | None = None,
    action_spec: ActionSpec | None = None,
    rollout_step_timeout_sec: float | int | None = None,
    rollout_failure_retries: int = 1,
    max_episodes_rendered_per_task: int = 0,
    videos_dir: Path | None = None,
    phase_heartbeat: Callable[[str], None] | None = None,
    progress_heartbeat: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate with one persistent async vec pool using lane-affine task quotas."""
    num_eval_per_task = require_positive_int(
        num_eval_per_task, name="num_eval_per_task"
    )
    num_parallel_env = require_positive_int(
        int(eval_pool.num_envs), name="eval_pool.num_envs"
    )
    retry_policy = RolloutFailureRetryPolicy(max_retries=rollout_failure_retries)

    if isinstance(policy, nn.Module) and hasattr(policy, "select_action"):
        eval_policy = policy
    else:
        eval_policy = LocalPolicyAdapter(
            policy,
            device=device,
            policy_kwargs=policy_kwargs,
            action_spec=action_spec,
        )
    wave_runner = PooledRolloutWaveRunner(
        policy=eval_policy,
        retry_policy=retry_policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        rollout_step_timeout_sec=rollout_step_timeout_sec,
        phase_heartbeat=phase_heartbeat,
    )

    start = time.time()
    all_results: list[TaskEvalResult] = []
    rendered_per_task: dict[tuple[str, int], int] = defaultdict(int)
    task_keys: list[tuple[str, int]] = [
        (task_group, int(task_id)) for task_group, task_id in tasks
    ]
    if len(task_keys) == 0:
        return summarize_task_eval_results(start=start, all_results=all_results)
    total_episodes = len(task_keys) * int(num_eval_per_task)
    total_successes = 0
    total_lengths = 0

    task_index: dict[tuple[str, int], int] = {
        task_key: idx for idx, task_key in enumerate(task_keys)
    }
    remaining: dict[tuple[str, int], int] = {
        task_key: int(num_eval_per_task) for task_key in task_keys
    }
    next_eval_idx: dict[tuple[str, int], int] = dict.fromkeys(task_keys, 0)

    lane_task: list[tuple[str, int] | None] = [None] * num_parallel_env
    lane_last_episode_index: list[int] = [0] * num_parallel_env
    lane_idle: list[bool] = [False] * num_parallel_env
    assign_cursor = 0
    total_waves = max(1, math.ceil(total_episodes / num_parallel_env))
    wave_idx = 0
    progress_cm = tqdm(
        total=total_episodes,
        desc="Eval",
        unit="ep",
        dynamic_ncols=True,
    )

    def _pick_next_unfinished_task() -> tuple[str, int] | None:
        nonlocal assign_cursor
        n_tasks = len(task_keys)
        for _ in range(n_tasks):
            candidate = task_keys[assign_cursor]
            assign_cursor = (assign_cursor + 1) % n_tasks
            if remaining[candidate] > 0:
                return candidate
        return None

    with progress_cm as pbar:
        pbar.set_postfix(
            succ_rate="0.0%",
            avg_len="0.0",
            ep_s="0.00",
            refresh=False,
        )
        while any(v > 0 for v in remaining.values()):
            wave_idx += 1
            if phase_heartbeat is not None:
                phase_heartbeat(f"wave_boundary idx={wave_idx} total={total_waves}")
            lane_jobs: list[EvalLaneJob | None] = []
            for lane_idx in range(num_parallel_env):
                previous_task = lane_task[lane_idx]
                current_task = previous_task
                if current_task is None or remaining[current_task] <= 0:
                    replacement_task = _pick_next_unfinished_task()
                    if replacement_task is not None:
                        current_task = replacement_task
                        lane_task[lane_idx] = replacement_task
                        lane_idle[lane_idx] = False
                    else:
                        current_task = previous_task
                        if previous_task is not None and not lane_idle[lane_idx]:
                            lane_idle[lane_idx] = True

                if current_task is None or remaining[current_task] <= 0:
                    lane_jobs.append(None)
                    continue

                eval_idx = int(next_eval_idx[current_task])
                next_eval_idx[current_task] += 1
                remaining[current_task] -= 1
                task_group, task_id = current_task
                episode_index = eval_idx * len(task_keys) + int(
                    task_index[current_task]
                )
                lane_last_episode_index[lane_idx] = int(episode_index)
                lane_jobs.append(
                    EvalLaneJob(
                        str(task_group),
                        int(task_id),
                        int(eval_idx),
                        int(episode_index),
                    )
                )

            active_lane_indices = [
                i for i, job in enumerate(lane_jobs) if job is not None
            ]
            if len(active_lane_indices) == 0:
                break

            prepared_jobs: list[EvalLaneJob | None] = []
            for lane_idx, lane_job in enumerate(lane_jobs):
                if lane_job is not None:
                    prepared_jobs.append(lane_job)
                    continue
                hold_task = lane_task[lane_idx]
                if hold_task is None:
                    prepared_jobs.append(None)
                    continue
                hold_eval_idx = max(0, int(next_eval_idx[hold_task]) - 1)
                hold_episode_index = int(lane_last_episode_index[lane_idx])
                prepared_jobs.append(
                    EvalLaneJob(
                        hold_task[0],
                        hold_task[1],
                        hold_eval_idx,
                        hold_episode_index,
                    )
                )

            if phase_heartbeat is not None:
                phase_heartbeat("prepare_jobs_begin")
            eval_pool.prepare_jobs(prepared_jobs)
            if phase_heartbeat is not None:
                phase_heartbeat("prepare_jobs_end")
            if eval_pool.env_pool is None:
                raise RuntimeError(
                    "Eval pool prepare_jobs completed without initializing env_pool."
                )
            prepared_num_envs = int(eval_pool.env_pool.num_envs)
            if prepared_num_envs != num_parallel_env:
                raise RuntimeError(
                    "Eval pool lane count mismatch after prepare_jobs: "
                    f"handle has {num_parallel_env}, env_pool has {prepared_num_envs}; "
                    f"lane_context={lane_context(prepared_jobs)}"
                )

            seeds: list[int] | None = [] if start_seed is not None else None
            max_episodes_rendered_by_env: list[int] = []
            videos_dirs_by_env: list[Path | None] = []
            video_start_index_by_env: list[int] = []
            reserved_videos_in_wave: dict[tuple[str, int], int] = defaultdict(int)

            for lane_idx, prepared_job in enumerate(prepared_jobs):
                if prepared_job is None:
                    if seeds is not None:
                        seeds.append(int(start_seed) if start_seed is not None else 0)
                    max_episodes_rendered_by_env.append(0)
                    videos_dirs_by_env.append(None)
                    video_start_index_by_env.append(0)
                    continue

                task_group, task_id, _eval_idx, episode_index = prepared_job
                active_lane = lane_jobs[lane_idx] is not None
                task_key = (task_group, task_id)
                already_rendered = int(rendered_per_task[task_key])
                already_reserved = int(reserved_videos_in_wave[task_key])
                remaining_videos = max(
                    0,
                    int(max_episodes_rendered_per_task)
                    - already_rendered
                    - already_reserved,
                )
                task_videos_dir = (
                    videos_dir / f"{task_group}_{task_id}"
                    if videos_dir is not None
                    else None
                )

                if seeds is not None:
                    assert start_seed is not None
                    seeds.append(int(start_seed) + int(episode_index))
                should_render = 1 if (active_lane and remaining_videos > 0) else 0
                max_episodes_rendered_by_env.append(should_render)
                videos_dirs_by_env.append(task_videos_dir if active_lane else None)
                video_start_index_by_env.append(already_rendered + already_reserved)
                if should_render > 0:
                    reserved_videos_in_wave[task_key] += 1

            lane_results = wave_runner.run(
                eval_pool=eval_pool,
                prepared_jobs=prepared_jobs,
                num_parallel_env=num_parallel_env,
                wave_idx=wave_idx,
                total_waves=total_waves,
                seeds=seeds,
                render_plan=RolloutRenderPlan(
                    max_episodes_rendered_by_env=max_episodes_rendered_by_env,
                    videos_dirs_by_env=videos_dirs_by_env,
                    video_start_index_by_env=video_start_index_by_env,
                ),
            )
            if len(lane_results) != num_parallel_env:
                raise RuntimeError(
                    "Pooled sim rollout returned the wrong number of lane results: "
                    f"expected {num_parallel_env}, got {len(lane_results)}; "
                    f"lane_context={lane_context(prepared_jobs)}"
                )

            completed_in_wave = 0
            for lane_idx in active_lane_indices:
                active_job = lane_jobs[lane_idx]
                assert active_job is not None
                task_group, task_id, _eval_idx, _episode_index = active_job
                result = lane_results[lane_idx]
                result.task_group = task_group
                result.task_id = task_id
                all_results.append(result)
                rendered_per_task[(task_group, task_id)] += len(result.video_paths)
                total_successes += sum(1 for success in result.successes if success)
                total_lengths += sum(int(length) for length in result.lengths)
                completed_in_wave += len(result.successes)

            completed = max(1, len(all_results))
            elapsed = max(time.time() - start, 1e-6)
            pbar.set_postfix(
                succ_rate=f"{(100.0 * total_successes / completed):.1f}%",
                avg_len=f"{(total_lengths / completed):.1f}",
                ep_s=f"{(len(all_results) / elapsed):.2f}",
                refresh=False,
            )
            pbar.update(completed_in_wave)
            if progress_heartbeat is not None:
                progress_heartbeat(
                    f"pbar_tick done={len(all_results)}/{total_episodes}"
                )

    summary = summarize_task_eval_results(start=start, all_results=all_results)
    diagnostics_summary = getattr(eval_policy, "policy_diagnostics_summary", None)
    if callable(diagnostics_summary):
        policy_diagnostics = diagnostics_summary()
        if policy_diagnostics:
            summary["policy_diagnostics"] = policy_diagnostics
    return summary
