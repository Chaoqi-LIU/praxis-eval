from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class PointPolicyHandler:
    """Server-side policy handler for the toy point-reaching task."""

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = float(gain)

    def predict_action(
        self,
        observations: Sequence[Mapping[str, Any]],
        *,
        policy_kwargs: Mapping[str, Any] | None = None,
        episode_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        del episode_ids
        gain = float((policy_kwargs or {}).get("gain", self.gain))
        actions = []
        for observation in observations:
            state = np.asarray(observation["observation.state"], dtype=np.float32)
            target = np.asarray(observation["observation.goal"], dtype=np.float32)
            actions.append(np.clip((target - state) * gain, -1.0, 1.0))
        return np.stack(actions).astype(np.float32)

    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        del episode_ids

    def model_info(self) -> str:
        return f"PointPolicyHandler(gain={self.gain})"
