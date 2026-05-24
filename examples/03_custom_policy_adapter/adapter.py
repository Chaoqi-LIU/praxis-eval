from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from fake_model import FakePointModel

from praxis_eval import ActionSpec, Observation


class PointModelPolicyAdapter:
    """Adapt FakePointModel to the praxis_eval.Policy protocol."""

    def __init__(self, model: FakePointModel) -> None:
        self.model = model

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
        del episode_ids
        gain = float((policy_kwargs or {}).get("gain", 1.0))
        actions = []
        for observation in observations:
            state = np.asarray(observation["observation.state"], dtype=np.float32)
            target = np.asarray(observation["observation.goal"], dtype=np.float32)
            action = self.model.predict(
                task=str(observation["task"]),
                state=state,
                target=target,
            )
            actions.append(np.clip(action * gain, -1.0, 1.0))

        batched = np.stack(actions).astype(np.float32)
        if action_spec is not None:
            return action_spec.validate_batch(batched, batch_size=len(observations))
        return batched
