# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Local in-process policy adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from praxis_eval.contracts import ActionSpec
from praxis_eval.policies.actions import normalize_batched_action
from praxis_eval.types import Observation


class LocalPolicy:
    """Adapter for a local callable or object with an ``act`` method."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            if episode_ids is None:
                reset()
            else:
                reset(episode_ids=episode_ids)

    def act(
        self,
        observations: Sequence[Observation],
        *,
        action_spec: ActionSpec | None = None,
        policy_kwargs: Mapping[str, Any] | None = None,
        episode_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        if len(observations) < 1:
            raise ValueError("observations must be non-empty")
        policy_kwargs = dict(policy_kwargs or {})
        if hasattr(self.policy, "act"):
            kwargs = {"policy_kwargs": policy_kwargs}
            if episode_ids is not None:
                kwargs["episode_ids"] = episode_ids
            action = self.policy.act(
                observations,
                **kwargs,
            )
        elif callable(self.policy):
            kwargs = {"policy_kwargs": policy_kwargs}
            if episode_ids is not None:
                kwargs["episode_ids"] = episode_ids
            action = self.policy(
                observations,
                **kwargs,
            )
        else:
            raise TypeError("LocalPolicy requires a callable or object with act().")
        return normalize_batched_action(
            action,
            batch_size=len(observations),
            action_spec=action_spec,
        )
