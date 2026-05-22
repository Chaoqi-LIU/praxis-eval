# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Metrics and aggregation utilities for simulation evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TaskEvalResult:
    task_group: str
    task_id: int
    task_description: str
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    lengths: list[int]
    video_paths: list[str]


@dataclass
class _TaskAccumulator:
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    lengths: list[int]
    task_description: str
    video_paths: list[str]


@dataclass
class _GroupAccumulator:
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    lengths: list[int]


def summarize_task_eval_results(
    *, start: float, all_results: list[TaskEvalResult]
) -> dict[str, Any]:
    per_task: dict[str, dict[str, float | str | list[str]]] = {}
    per_task_acc: dict[str, _TaskAccumulator] = {}
    per_group_acc: dict[str, _GroupAccumulator] = {}

    all_sum_rewards: list[float] = []
    all_max_rewards: list[float] = []
    all_successes: list[bool] = []
    all_lengths: list[int] = []
    all_video_paths: list[str] = []

    for result in sorted(all_results, key=lambda r: (r.task_group, r.task_id)):
        key = f"{result.task_group}/{result.task_id}"
        acc = per_task_acc.setdefault(
            key,
            _TaskAccumulator([], [], [], [], "", []),
        )
        acc.sum_rewards.extend(result.sum_rewards)
        acc.max_rewards.extend(result.max_rewards)
        acc.successes.extend(result.successes)
        acc.lengths.extend(result.lengths)
        acc.video_paths.extend(result.video_paths)
        if not acc.task_description and result.task_description:
            acc.task_description = result.task_description

        group_acc = per_group_acc.setdefault(
            result.task_group, _GroupAccumulator([], [], [], [])
        )
        group_acc.sum_rewards.extend(result.sum_rewards)
        group_acc.max_rewards.extend(result.max_rewards)
        group_acc.successes.extend(result.successes)
        group_acc.lengths.extend(result.lengths)

        all_sum_rewards.extend(result.sum_rewards)
        all_max_rewards.extend(result.max_rewards)
        all_successes.extend(result.successes)
        all_lengths.extend(result.lengths)
        all_video_paths.extend(result.video_paths)

    for key, acc in per_task_acc.items():
        successes = acc.successes
        lengths = acc.lengths
        sum_rewards = acc.sum_rewards
        per_task[key] = {
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "avg_episode_length": float(np.mean(lengths)) if lengths else 0.0,
            "avg_reward": float(np.mean(sum_rewards)) if sum_rewards else 0.0,
            "n_episodes": float(len(successes)),
            "task_description": acc.task_description,
            "video_paths": list(acc.video_paths),
        }

    per_group: dict[str, dict[str, float]] = {}
    for group, acc in per_group_acc.items():
        group_success = float(np.mean(acc.successes)) if acc.successes else 0.0
        per_group[group] = {
            "success_rate": group_success,
            "avg_episode_length": float(np.mean(acc.lengths)) if acc.lengths else 0.0,
            "avg_reward": float(np.mean(acc.sum_rewards)) if acc.sum_rewards else 0.0,
            "avg_sum_reward": float(np.mean(acc.sum_rewards))
            if acc.sum_rewards
            else 0.0,
            "avg_max_reward": float(np.mean(acc.max_rewards))
            if acc.max_rewards
            else 0.0,
            "n_episodes": float(len(acc.successes)),
        }

    elapsed = time.time() - start
    overall = {
        "success_rate": float(np.mean(all_successes)) if all_successes else 0.0,
        "avg_episode_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "avg_reward": float(np.mean(all_sum_rewards)) if all_sum_rewards else 0.0,
        "avg_sum_reward": float(np.mean(all_sum_rewards)) if all_sum_rewards else 0.0,
        "avg_max_reward": float(np.mean(all_max_rewards)) if all_max_rewards else 0.0,
        "n_episodes": float(len(all_successes)),
        "eval_s": elapsed,
        "eval_ep_s": elapsed / max(1, len(all_successes)),
        "video_paths": all_video_paths,
    }

    return {
        "overall": overall,
        "per_group": per_group,
        "per_task": per_task,
    }
