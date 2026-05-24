from __future__ import annotations

import numpy as np


class FakePointModel:
    """Small stand-in for a checkpoint-backed policy model."""

    def predict(
        self, *, task: str, state: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        del task
        return np.clip(target - state, -1.0, 1.0).astype(np.float32)
