# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Policy inference diagnostics collected during simulation evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from functools import cache
from numbers import Real
from typing import Any

import numpy as np

NON_AGGREGATED_DIAGNOSTIC_KEYS = frozenset({"pred_action_token_ids"})


class InferenceDiagnosticsAccumulator:
    """Accumulate numeric policy metadata emitted during rollout inference."""

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)

    def add(self, info: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None) -> None:
        if not info:
            return
        if not isinstance(info, Mapping):
            for item in info:
                self.add(item)
            return
        for key, value in info.items():
            if key in NON_AGGREGATED_DIAGNOSTIC_KEYS:
                continue
            values = list(_iter_numeric_values(value))
            if values:
                self._values[str(key)].extend(values)

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key, values in sorted(self._values.items()):
            if not values:
                continue
            out[key] = _summarize_numeric_values(values)
        return out


def _iter_numeric_values(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        yield float(value)
        return
    torch_tensor_type = _torch_tensor_type()
    if torch_tensor_type is not None and isinstance(value, torch_tensor_type):
        yield from (float(v) for v in value.detach().cpu().reshape(-1).tolist())
        return
    if isinstance(value, np.ndarray):
        yield from (float(v) for v in value.reshape(-1).tolist())
        return
    if _is_real_number(value):
        yield float(value)
        return
    if isinstance(value, Mapping):
        return
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _iter_numeric_values(item)


def _summarize_numeric_values(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    summary: dict[str, Any] = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    if _all_integral(values):
        unique, counts = np.unique(arr.astype(np.int64), return_counts=True)
        total = float(arr.size)
        hist = {str(int(k)): int(v) for k, v in zip(unique, counts, strict=True)}
        frac = {
            str(int(k)): float(int(v) / total)
            for k, v in zip(unique, counts, strict=True)
        }
        summary["hist"] = hist
        summary["frac"] = frac
    return summary


def _all_integral(values: list[float]) -> bool:
    return all(float(v).is_integer() for v in values)


def _is_real_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


@cache
def _torch_tensor_type() -> type[Any] | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.Tensor
