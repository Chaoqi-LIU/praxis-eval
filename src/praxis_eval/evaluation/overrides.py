# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for evaluation-specific Hydra override handling."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

_DECLARED_EVAL_POLICY_KWARGS = frozenset(
    {
        "decode_keep_k",
        "decode_temperature",
        "decode_top_k",
    }
)

EvalTargetInferer = Callable[[str], tuple[str, str]]


def normalize_eval_overrides(eval_overrides: list[str]) -> list[str]:
    """Normalize eval-only Hydra overrides for launcher and monitor callers."""
    normalized: list[str] = []
    for override in eval_overrides:
        text = str(override)
        if text.startswith("+"):
            normalized.append(text)
            continue
        if text.startswith("eval.policy_kwargs."):
            key_path = text.split("=", 1)[0].removeprefix("eval.policy_kwargs.")
            key = key_path.split(".", 1)[0]
            if key in _DECLARED_EVAL_POLICY_KWARGS:
                normalized.append(text)
                continue
            normalized.append(f"+{text}")
            continue
        normalized.append(text)
    return normalized


def build_eval_overrides_from_train_config(
    train_config_path: Path,
    *,
    num_eval_per_task: int,
    num_parallel_env: int,
    record_episodes_per_task: int,
    infer_env_from_dataset: EvalTargetInferer | None = None,
) -> list[str]:
    """Build ``praxis-eval`` Hydra overrides from a checkpoint train config."""
    cfg = _load_train_config(train_config_path)

    policy_name = _optional_string(_select_config_value(cfg, "policy.name"))
    env_type = _optional_string(_select_config_value(cfg, "env.type"))
    env_task = _optional_string(_select_config_value(cfg, "env.task"))
    device = _select_config_value(cfg, "device")

    if not env_type or not env_task:
        dataset_name = _optional_string(_select_config_value(cfg, "dataset.name"))
        if not dataset_name:
            raise ValueError(
                f"train_config missing env fields and dataset.name: {train_config_path}"
            )
        inferer = infer_env_from_dataset or _default_eval_target_inferer
        env_type, env_task = inferer(dataset_name)
        if not env_type:
            raise ValueError(
                f"train_config missing env fields and cannot infer env type "
                f"from dataset {dataset_name!r}: {train_config_path}"
            )

    if not policy_name:
        raise ValueError(
            f"train_config missing required policy.name: {train_config_path}"
        )

    env_task_override = (
        f"env.task='{env_task}'" if "," in env_task else f"env.task={env_task}"
    )
    overrides = [
        f"policy={policy_name}",
        f"env.type={env_type}",
        env_task_override,
        "eval.mode=local_inproc",
        f"eval.num_eval_per_task={num_eval_per_task}",
        f"eval.num_parallel_env={num_parallel_env}",
        f"eval.record_episodes_per_task={record_episodes_per_task}",
    ]
    if device is not None:
        overrides.append(f"device={device}")
    task_ids = _select_config_value(cfg, "env.task_ids")
    if task_ids is not None:
        overrides.append(f"env.task_ids={task_ids}")
    return overrides


def _load_train_config(train_config_path: Path) -> Any:
    cfg = OmegaConf.load(train_config_path)
    if not OmegaConf.is_config(cfg):
        raise TypeError(f"train_config must be a mapping, got {type(cfg)!r}")
    return cfg


def _select_config_value(cfg: Any, path: str) -> Any:
    if OmegaConf.select(cfg, path) is None:
        return None
    value = OmegaConf.select(cfg, path)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _default_eval_target_inferer(dataset_name: str) -> tuple[str, str]:
    from praxis_eval.envs.factory import infer_eval_env_target

    return infer_eval_env_target(dataset_name)
