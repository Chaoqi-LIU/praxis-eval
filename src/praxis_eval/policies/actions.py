# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared action normalization helpers for policy adapters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from praxis_eval.contracts import ActionSpec


def normalize_batched_action(
    action: np.ndarray | Sequence[np.ndarray],
    *,
    batch_size: int,
    action_spec: ActionSpec | None,
) -> np.ndarray:
    """Normalize policy output into a batched action array and validate it."""
    arr = np.asarray(action)
    if batch_size < 1:
        raise ValueError("observations must be non-empty")
    if action_spec is not None:
        single_shape = (
            tuple(action_spec.shape) if action_spec.shape is not None else None
        )
        if (
            single_shape is not None
            and tuple(arr.shape) == single_shape
            and batch_size == 1
        ):
            arr = arr.reshape((1, *single_shape))
        return action_spec.validate_batch(arr, batch_size=batch_size)
    if arr.ndim == 0:
        raise ValueError("Batched action must have at least one dimension.")
    if batch_size == 1 and arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[0] != batch_size:
        raise ValueError(
            f"Batched action leading dim {arr.shape[0]} does not match {batch_size}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Action contains non-finite values.")
    return arr
