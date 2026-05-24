from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from praxis_eval import ActionSpec, Observation


class RandomPolicy:
    """Policy that samples valid actions from the eval-provided ActionSpec."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        del episode_ids

    def act(
        self,
        observations: Sequence[Observation],
        *,
        action_spec: ActionSpec | None = None,
        policy_kwargs: Mapping[str, Any] | None = None,
        episode_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        del policy_kwargs, episode_ids
        if action_spec is None or action_spec.shape is None:
            raise ValueError("RandomPolicy needs an ActionSpec with a fixed shape.")

        low = -1.0 if action_spec.minimum is None else action_spec.minimum
        high = 1.0 if action_spec.maximum is None else action_spec.maximum
        return self.rng.uniform(
            low=low,
            high=high,
            size=(len(observations), *action_spec.shape),
        ).astype(action_spec.dtype)
