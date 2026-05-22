# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared public types for eval and policy adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np

from praxis_eval.contracts import ActionSpec

NumpyScalar: TypeAlias = np.bool_ | np.integer[Any] | np.floating[Any]
ObservationValue: TypeAlias = np.ndarray | str | bool | int | float | NumpyScalar
Observation: TypeAlias = Mapping[str, ObservationValue]


@dataclass(frozen=True)
class EvalRuntimeHooks:
    """Optional runtime callbacks for hosts that monitor evaluator progress."""

    phase_heartbeat: Callable[[str], None] | None = None
    progress_heartbeat: Callable[[str], None] | None = None


@dataclass(frozen=True)
class EvalConfig:
    """Generic evaluation configuration shared across env drivers."""

    num_eval_per_task: int
    output_dir: str | Path
    task: str | None = None
    task_ids: tuple[int, ...] | None = None
    num_parallel_env: int = 1
    seed: int = 42
    record_episodes_per_task: int = 0
    step_timeout_sec: float | None = None
    rollout_failure_retries: int = 1
    debug_verbose: bool = False
    policy_kwargs: Mapping[str, Any] = field(default_factory=dict)
    env_kwargs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_hooks: EvalRuntimeHooks = field(default_factory=EvalRuntimeHooks)


@dataclass(frozen=True)
class EvalResult:
    """Evaluation result returned by env drivers."""

    overall: Mapping[str, Any]
    per_task: Mapping[str, Mapping[str, Any]]
    per_group: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    """Policy adapter protocol used by eval drivers."""

    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        """Reset policy rollout state before an episode or task wave."""
        ...

    def act(
        self,
        observations: Sequence[Observation],
        *,
        action_spec: ActionSpec | None = None,
        policy_kwargs: Mapping[str, Any] | None = None,
        episode_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return a batched action array for ``observations``.

        ``action_spec`` is eval-side validation metadata. Policy adapters should
        not require wrapped policies to consume it.
        """
        ...
