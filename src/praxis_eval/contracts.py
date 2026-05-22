# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Documented observation and action contracts for eval environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObservationKey:
    """One observation key emitted by an environment."""

    key: str
    dtype: str
    shape: tuple[int | str, ...] | None = None
    description: str = ""


@dataclass(frozen=True)
class ActionSpec:
    """Action contract expected by an environment."""

    shape: tuple[int, ...] | None
    dtype: str = "float32"
    minimum: float | None = None
    maximum: float | None = None
    convention: str = ""
    description: str = ""

    def validate(self, action: np.ndarray) -> np.ndarray:
        """Validate and return ``action`` as a numpy array."""
        arr = np.asarray(action)
        if self.shape is not None and tuple(arr.shape) != tuple(self.shape):
            raise ValueError(
                f"Action shape {tuple(arr.shape)} does not match expected {self.shape}."
            )
        expected_dtype = np.dtype(self.dtype)
        if arr.dtype != expected_dtype:
            raise TypeError(
                f"Action dtype {arr.dtype} does not match {expected_dtype}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Action contains non-finite values.")
        if self.minimum is not None and np.any(arr < self.minimum):
            raise ValueError(f"Action contains values below {self.minimum}.")
        if self.maximum is not None and np.any(arr > self.maximum):
            raise ValueError(f"Action contains values above {self.maximum}.")
        return arr

    def validate_batch(self, actions: np.ndarray, *, batch_size: int) -> np.ndarray:
        """Validate and return a batched action array."""
        arr = np.asarray(actions)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.shape is None:
            if int(arr.shape[0]) != int(batch_size):
                raise ValueError(
                    f"Batched action leading dim {arr.shape[0]} does not match {batch_size}."
                )
            for row in arr:
                self.validate(row)
            return arr
        expected = (int(batch_size), *tuple(self.shape))
        if tuple(arr.shape) != expected:
            raise ValueError(
                f"Batched action shape {tuple(arr.shape)} does not match {expected}."
            )
        for row in arr:
            self.validate(row)
        return arr


@dataclass(frozen=True)
class EnvContract:
    """Observation/action contract for one eval environment family."""

    env_type: str
    observation_keys: tuple[ObservationKey, ...]
    action: ActionSpec
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
