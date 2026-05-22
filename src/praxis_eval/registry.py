# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluation-driver registry."""

from __future__ import annotations

from typing import Protocol

from praxis_eval.contracts import EnvContract
from praxis_eval.types import EvalConfig, EvalResult, Policy


class EvalDriver(Protocol):
    """Env-family evaluation driver interface."""

    @property
    def contract(self) -> EnvContract:
        """Return the documented observation/action contract."""
        ...

    def evaluate(self, *, policy: Policy, config: EvalConfig) -> EvalResult:
        """Run evaluation and return metrics/artifact metadata."""
        ...


_DRIVERS: dict[str, EvalDriver] = {}
_BUILTINS_REGISTERED = False


def register_driver(name: str, driver: EvalDriver, *, replace: bool = True) -> None:
    """Register an env-family evaluation driver."""
    key = _normalize_name(name)
    if not key:
        raise ValueError("driver name must be non-empty")
    if not replace and key in _DRIVERS:
        return
    _DRIVERS[key] = driver


def get_driver(name: str) -> EvalDriver:
    """Return a registered evaluation driver."""
    _ensure_builtin_drivers_registered()
    key = _normalize_name(name)
    try:
        return _DRIVERS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_DRIVERS)) or "(none)"
        raise ValueError(
            f"Unknown eval driver {name!r}. Available: {available}"
        ) from exc


def available_drivers() -> tuple[str, ...]:
    """Return registered driver names."""
    _ensure_builtin_drivers_registered()
    return tuple(sorted(_DRIVERS))


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _ensure_builtin_drivers_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True
    from praxis_eval.envs.builtins import register_builtin_contract_drivers

    register_builtin_contract_drivers()
