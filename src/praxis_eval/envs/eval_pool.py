# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Generic persistent-eval pool handle and lane-job types."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import gymnasium as gym
else:
    gym = Any


class EvalLaneJob(NamedTuple):
    """One eval assignment for one vector-env lane."""

    task_group: str
    task_id: int
    eval_idx: int
    episode_index: int


class EvalPoolHandle:
    """Generic wrapper around a persistent vector env plus job-preparation logic."""

    def __init__(
        self,
        *,
        env_pool: gym.vector.VectorEnv | None,
        prepare_jobs: Callable[[list[EvalLaneJob | None]], None],
        num_envs: int | None = None,
    ) -> None:
        self.env_pool = env_pool
        if num_envs is None:
            if env_pool is None:
                raise ValueError("num_envs is required when env_pool is None.")
            num_envs = int(env_pool.num_envs)
        self._num_envs = int(num_envs)
        self._prepare_jobs = prepare_jobs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def prepare_jobs(self, lane_jobs: list[EvalLaneJob | None]) -> None:
        self._prepare_jobs(lane_jobs)

    def close(self, *, terminate: bool = False) -> None:
        env_pool = self.env_pool
        if env_pool is None:
            return
        try:
            env_pool.close(terminate=terminate)
        finally:
            self.env_pool = None
