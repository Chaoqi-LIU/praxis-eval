# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Policy adapters used by simulation evaluation runtime."""

from __future__ import annotations

from collections.abc import Sequence
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
            action_spec=self.action_spec,
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
    """Split a batch-first observation dict into per-example numpy observations."""
    batch_size = _infer_batch_size(batch)
    observations: list[dict[str, ObservationValue]] = [{} for _ in range(batch_size)]

    for key, value in batch.items():
        if key in _ROLLOUT_BOOKKEEPING_KEYS or value is None:
            continue

        if isinstance(value, Tensor):
            array = value.detach().cpu().numpy()
            if int(array.shape[0]) != batch_size:
                raise ValueError(
                    f"Observation batch for key {key!r} has leading dim {array.shape[0]}, "
                    f"expected {batch_size}."
                )
            for index in range(batch_size):
                observations[index][key] = np.asarray(array[index])
            continue

        if isinstance(value, np.ndarray):
            if int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"Observation batch for key {key!r} has leading dim {value.shape[0]}, "
                    f"expected {batch_size}."
                )
            for index in range(batch_size):
                observations[index][key] = np.asarray(value[index])
            continue

        if isinstance(value, _SCALAR_OBSERVATION_TYPES):
            if batch_size != 1:
                raise TypeError(
                    f"Observation key {key!r} is a scalar but batch size is {batch_size}."
                )
            observations[0][key] = (
                value.item() if isinstance(value, np.generic) else value
            )
            continue

        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if len(value) != batch_size:
                raise ValueError(
                    f"Observation batch for key {key!r} has length {len(value)}, "
                    f"expected {batch_size}."
                )
            if not all(isinstance(item, _SCALAR_OBSERVATION_TYPES) for item in value):
                raise TypeError(
                    f"Observation key {key!r} must be tensor-like or a sequence of scalars, "
                    f"got sequence element types {[type(item).__name__ for item in value]}."
                )
            for index, item in enumerate(value):
                observations[index][key] = (
                    item.item() if isinstance(item, np.generic) else item
                )
            continue

        raise TypeError(
            f"Unsupported observation value type for key {key!r}: {type(value)!r}."
        )

    return observations


def _infer_batch_size(batch: dict[str, Any]) -> int:
    for key, value in batch.items():
        if key in _ROLLOUT_BOOKKEEPING_KEYS or value is None:
            continue

        if isinstance(value, Tensor):
            if value.dim() == 0:
                raise ValueError(
                    "Batched remote inference requires tensors with a batch dimension."
                )
            return int(value.shape[0])
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                raise ValueError(
                    "Batched remote inference requires numpy arrays with a batch dimension."
                )
            return int(value.shape[0])
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return len(value)
        if isinstance(value, _SCALAR_OBSERVATION_TYPES):
            return 1
    return 1
