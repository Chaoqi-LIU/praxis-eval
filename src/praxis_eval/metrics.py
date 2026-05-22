# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Metrics aggregation for evaluation drivers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeResult:
    """One completed rollout episode."""

    task_key: str
    success: bool
    episode_length: int
    reward: float = 0.0
    max_reward: float = 0.0
    video_paths: list[str] = field(default_factory=list)


def summarize_episodes(
    episodes: list[EpisodeResult],
    *,
    start_time: float | None = None,
) -> dict[str, dict[str, float | list[str]]]:
    """Aggregate episode metrics into overall and per-task summaries."""
    start = time.time() if start_time is None else float(start_time)
    elapsed = max(time.time() - start, 0.0)
    per_task: dict[str, list[EpisodeResult]] = {}
    for episode in episodes:
        per_task.setdefault(episode.task_key, []).append(episode)

    task_summary = {
        key: _summarize_group(items, elapsed_s=None) for key, items in per_task.items()
    }
    overall = _summarize_group(episodes, elapsed_s=elapsed)
    return {
        "overall": overall,
        "per_task": task_summary,
    }


def _summarize_group(
    episodes: list[EpisodeResult],
    *,
    elapsed_s: float | None,
) -> dict[str, float | list[str]]:
    successes = [episode.success for episode in episodes]
    lengths = [episode.episode_length for episode in episodes]
    rewards = [episode.reward for episode in episodes]
    max_rewards = [episode.max_reward for episode in episodes]
    videos = [path for episode in episodes for path in episode.video_paths]
    result: dict[str, float | list[str]] = {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_max_reward": float(np.mean(max_rewards)) if max_rewards else 0.0,
        "n_episodes": float(len(episodes)),
        "video_paths": videos,
    }
    if elapsed_s is not None:
        result["eval_s"] = float(elapsed_s)
        result["eval_ep_s"] = float(elapsed_s) / max(1, len(episodes))
    return result
