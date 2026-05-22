# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight config parsing helpers shared by eval entrypoints."""

from __future__ import annotations

from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def require_positive_int(value: int, *, name: str) -> int:
    """Validate positive integer config values used by eval schedulers."""
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def resolve_nonnegative_int(
    section: DictConfig | dict[str, Any],
    *,
    key: str,
    label: str,
    default: int = 0,
) -> int:
    """Resolve a non-negative integer from a config section."""
    raw_value = _select_config_value(section, key)
    value = int(default if raw_value is None else raw_value)
    if value < 0:
        raise ValueError(f"{label} must be >= 0, got {value}")
    return value


def resolve_optional_positive_timeout_sec(
    section: DictConfig | dict[str, Any],
    *,
    key: str = "step_timeout_sec",
) -> float | None:
    """Resolve optional timeout seconds where null or non-positive disables it."""
    raw_value = _select_config_value(section, key)
    if raw_value is None:
        return None
    value = float(raw_value)
    return value if value > 0 else None


def resolve_policy_kwargs(
    section: DictConfig | dict[str, Any],
    *,
    key: str = "policy_kwargs",
    label: str,
) -> dict[str, Any]:
    """Resolve policy kwargs forwarded to eval-time policy inference."""
    raw_cfg = _select_config_value(section, key)
    if raw_cfg is None:
        return {}
    raw = (
        OmegaConf.to_container(raw_cfg, resolve=True)
        if isinstance(raw_cfg, DictConfig)
        else raw_cfg
    )
    if not isinstance(raw, dict):
        raise TypeError(
            f"{label} must resolve to a dict when provided, got {type(raw)!r}."
        )
    return {
        str(key): value
        for key, value in cast(dict[Any, Any], raw).items()
        if value is not None
    }


def env_to_plain_dict(env_cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Resolve an eval env config section into a plain string-keyed dict."""
    if isinstance(env_cfg, DictConfig):
        data = OmegaConf.to_container(env_cfg, resolve=True)
    else:
        data = dict(env_cfg)
    if not isinstance(data, dict):
        raise TypeError(f"env config must resolve to dict, got {type(data)!r}")
    return {str(key): value for key, value in data.items()}


def env_type_from_cfg(env_cfg: DictConfig | dict[str, Any]) -> str:
    """Return the required normalized eval env type selector."""
    env_type = env_to_plain_dict(env_cfg).get("type")
    if env_type is None:
        raise ValueError("env config must include `type`.")
    return str(env_type).strip().lower()


def optional_env_task(env_cfg: DictConfig | dict[str, Any]) -> str | None:
    """Return the optional eval env task selector."""
    value = env_to_plain_dict(env_cfg).get("task")
    return None if value is None else str(value)


def optional_env_task_ids(
    env_cfg: DictConfig | dict[str, Any],
) -> tuple[int, ...] | None:
    """Return optional eval env task ids as an immutable tuple."""
    value = env_to_plain_dict(env_cfg).get("task_ids")
    if value is None:
        return None
    return tuple(int(item) for item in value)


def env_kwargs_without_type_task(
    env_cfg: DictConfig | dict[str, Any],
) -> dict[str, Any]:
    """Return evaluator env kwargs after removing dispatch selectors."""
    data = env_to_plain_dict(env_cfg)
    for key in ("type", "task", "task_ids"):
        data.pop(key, None)
    return data


def _select_config_value(
    section: DictConfig | dict[str, Any],
    key: str,
) -> Any:
    if isinstance(section, DictConfig):
        return OmegaConf.select(section, key, default=None)
    return section.get(key)
