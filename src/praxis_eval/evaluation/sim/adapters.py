# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Policy adapters used by simulation evaluation runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from praxis_eval.contracts import ActionSpec
from praxis_eval.evaluation.sim.diagnostics import InferenceDiagnosticsAccumulator
from praxis_eval.policies.actions import normalize_batched_action
from praxis_eval.types import ObservationValue, Policy

_SCALAR_OBSERVATION_TYPES = (str, bool, int, float, np.generic)
_ROLLOUT_BOOKKEEPING_KEYS = frozenset({"action", "info"})


class LocalPolicyAdapter(nn.Module):
    """Adapter exposing LeRobot-style ``select_action`` for eval policies."""

    def __init__(
        self,
        policy: Policy,
        device: str | torch.device,
        *,
        policy_kwargs: dict[str, Any] | None = None,
        action_spec: ActionSpec | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.device = torch.device(device)
        self.policy_kwargs = dict(policy_kwargs or {})
        self.action_spec = action_spec
        self._diagnostics = InferenceDiagnosticsAccumulator()

    def forward(self, batch: dict[str, Tensor]):
        return self.select_action(batch)

    def reset(self) -> None:
        self.policy.reset()

    @torch.inference_mode()
    def select_action(self, batch: dict[str, Any]) -> Tensor:
        observations = _split_batched_observations(batch)
        action = self.policy.act(
            observations,
            action_spec=self.action_spec,
            policy_kwargs=self.policy_kwargs,
        )
        consume_info = getattr(self.policy, "consume_inference_info", None)
        if callable(consume_info):
            self._diagnostics.add(consume_info())
        action_array = normalize_batched_action(
            action,
            batch_size=len(observations),
            action_spec=_action_spec_without_bounds(self.action_spec),
        )
        action_tensor = torch.from_numpy(action_array).to(self.device)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        return action_tensor

    def policy_diagnostics_summary(self) -> dict[str, dict[str, Any]]:
        return self._diagnostics.summary()


def _split_batched_observations(
    batch: dict[str, Any],
) -> list[dict[str, ObservationValue]]:
    """Split a batch-first observation dict into per-example numpy observations.

    Scalar metadata is broadcast across all examples. Tensor, ndarray, and
    sequence values determine and validate the batch size.
    """
    batch_size = _infer_batch_size(batch)
    observations: list[dict[str, ObservationValue]] = [{} for _ in range(batch_size)]

    for key, value in batch.items():
        if key in _ROLLOUT_BOOKKEEPING_KEYS or value is None:
            continue

        if isinstance(value, Tensor):
            array = value.detach().cpu().numpy()
            _validate_batch_length(key, int(array.shape[0]), batch_size)
            for index in range(batch_size):
                observations[index][key] = np.asarray(array[index])
            continue

        if isinstance(value, np.ndarray):
            _validate_batch_length(key, int(value.shape[0]), batch_size)
            for index in range(batch_size):
                observations[index][key] = np.asarray(value[index])
            continue

        if isinstance(value, _SCALAR_OBSERVATION_TYPES):
            scalar = _as_python_scalar(value)
            for index in range(batch_size):
                observations[index][key] = scalar
            continue

        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            _validate_batch_length(key, len(value), batch_size)
            if not all(isinstance(item, _SCALAR_OBSERVATION_TYPES) for item in value):
                raise TypeError(
                    f"Observation key {key!r} must be tensor-like or a sequence of scalars, "
                    f"got sequence element types {[type(item).__name__ for item in value]}."
                )
            for index, item in enumerate(value):
                observations[index][key] = _as_python_scalar(item)
            continue

        raise TypeError(
            f"Unsupported observation value type for key {key!r}: {type(value)!r}."
        )

    return observations


def _infer_batch_size(batch: dict[str, Any]) -> int:
    sizes: dict[str, int] = {}
    for key, value in batch.items():
        if key in _ROLLOUT_BOOKKEEPING_KEYS or value is None:
            continue

        batch_size = _batch_size_constraint(key, value)
        if batch_size is not None:
            sizes[key] = batch_size

    if not sizes:
        return 1
    unique_sizes = set(sizes.values())
    if len(unique_sizes) != 1:
        details = ", ".join(f"{key}={size}" for key, size in sorted(sizes.items()))
        raise ValueError(f"Inconsistent observation batch sizes: {details}.")
    return next(iter(unique_sizes))


def _batch_size_constraint(key: str, value: Any) -> int | None:
    if isinstance(value, Tensor):
        if value.dim() == 0:
            raise ValueError(
                f"Observation key {key!r} is a zero-dimensional tensor; "
                "batched remote inference requires tensors with a batch dimension."
            )
        return int(value.shape[0])
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise ValueError(
                f"Observation key {key!r} is a zero-dimensional numpy array; "
                "batched remote inference requires arrays with a batch dimension."
            )
        return int(value.shape[0])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    if isinstance(value, _SCALAR_OBSERVATION_TYPES):
        return None
    return None


def _validate_batch_length(key: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(
            f"Observation batch for key {key!r} has length {actual}, expected {expected}."
        )


def _as_python_scalar(value: Any) -> ObservationValue:
    return value.item() if isinstance(value, np.generic) else value


def _action_spec_without_bounds(action_spec: ActionSpec | None) -> ActionSpec | None:
    if action_spec is None:
        return None
    if action_spec.minimum is None and action_spec.maximum is None:
        return action_spec
    return replace(action_spec, minimum=None, maximum=None)
