# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Environment config/env factory aligned with LeRobot."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from praxis_eval.envs.eval_pool import EvalPoolHandle
else:
    EvalPoolHandle = Any

_ENV_CONFIG_REGISTRY: dict[str, str] = {
    "aloha": "lerobot.envs.configs:AlohaEnv",
}

# Env types that are not registered in lerobot and must be instantiated directly.
_DIRECT_ENV_TYPES: frozenset[str] = frozenset(
    {
        "libero",
        "metaworld",
        "mshab",
        "robocasa",
        "robocasa_gr1",
        "robomimic",
        "simpler",
    }
)
AsyncEnvBuilder = Callable[[Any, int], Any]
EnvBuilder = Callable[[Any, int, bool], Any]
EvalPoolBuilder = Callable[[Any, list[tuple[str, int]], int, bool], EvalPoolHandle]
TaskLister = Callable[[dict[str, Any], Any, bool], list[tuple[str, int]]]
EvalTargetInferer = Callable[[str], tuple[str, str] | None]
_ENV_BUILDER_REGISTRY: dict[str, EnvBuilder | str] = {}
_ASYNC_ENV_BUILDER_REGISTRY: dict[str, AsyncEnvBuilder | str] = {}
_EVAL_POOL_BUILDER_REGISTRY: dict[str, EvalPoolBuilder | str] = {}
_TASK_LISTER_REGISTRY: dict[str, TaskLister | str] = {}
_EVAL_TARGET_INFERER_REGISTRY: dict[str, EvalTargetInferer | str] = {}


def register_env_config(name: str, import_path: str) -> None:
    """Register env config class import path.

    Args:
        name: Registry key used by ``cfg.env.type``.
        import_path: ``"module.path:ClassName"``.
    """
    key = _normalize_type_name(name)
    if ":" not in import_path:
        raise ValueError(
            f"Invalid import path {import_path!r}. Expected 'module.path:ClassName'."
        )
    _ENV_CONFIG_REGISTRY[key] = import_path


def register_async_env_builder(name: str, builder: AsyncEnvBuilder | str) -> None:
    """Register a custom async env builder for an env type."""
    _ASYNC_ENV_BUILDER_REGISTRY[_normalize_type_name(name)] = builder


def register_env_builder(name: str, builder: EnvBuilder | str) -> None:
    """Register a local env builder for an env type."""
    _ENV_BUILDER_REGISTRY[_normalize_type_name(name)] = builder


def register_eval_pool_builder(name: str, builder: EvalPoolBuilder | str) -> None:
    """Register a custom persistent-eval async-pool builder for an env type."""
    _EVAL_POOL_BUILDER_REGISTRY[_normalize_type_name(name)] = builder


def register_task_lister(name: str, lister: TaskLister | str) -> None:
    """Register a task lister for an env type."""
    _TASK_LISTER_REGISTRY[_normalize_type_name(name)] = lister


def register_eval_target_inferer(name: str, inferer: EvalTargetInferer | str) -> None:
    """Register dataset-name to eval-target inference for an env type."""
    _EVAL_TARGET_INFERER_REGISTRY[_normalize_type_name(name)] = inferer


def available_env_types() -> tuple[str, ...]:
    """Return registered environment type names."""
    return tuple(sorted(_ENV_CONFIG_REGISTRY))


def available_async_env_types() -> tuple[str, ...]:
    """Return env type names with custom async builders."""
    return tuple(sorted(_ASYNC_ENV_BUILDER_REGISTRY))


def available_eval_pool_env_types() -> tuple[str, ...]:
    """Return env type names with persistent eval pool builders."""
    return tuple(sorted(_EVAL_POOL_BUILDER_REGISTRY))


def build_env_config(env_cfg: DictConfig | dict[str, Any]):
    """Build an evaluator env config from a plain or OmegaConf mapping."""
    data = _to_plain_dict(env_cfg)
    if "type" not in data:
        raise ValueError("env config must include `type`.")
    env_type = _normalize_type_name(data.pop("type"))

    cfg_cls = _get_env_config_class(env_type)
    allowed_keys = (
        {f.name for f in fields(cfg_cls)} if is_dataclass(cfg_cls) else set(data.keys())
    )
    filtered = {k: v for k, v in data.items() if k in allowed_keys}

    if env_type in _DIRECT_ENV_TYPES:
        return cfg_cls(**filtered)
    return _make_lerobot_env_config(env_type, **filtered)


def make_env(
    env_cfg: DictConfig | dict[str, Any],
    *,
    n_envs: int = 1,
    use_async_envs: bool = False,
):
    """Create LeRobot vectorized env dict for the given config."""
    cfg_obj = build_env_config(env_cfg)
    env_type = _env_type_from_cfg(cfg_obj)

    if use_async_envs:
        async_builder = _ASYNC_ENV_BUILDER_REGISTRY.get(env_type)
        if async_builder is not None:
            return _load_callable(async_builder)(cfg_obj, n_envs)

    env_builder = _ENV_BUILDER_REGISTRY.get(env_type)
    if env_builder is not None:
        return _load_callable(env_builder)(cfg_obj, n_envs, use_async_envs)

    if env_type in _DIRECT_ENV_TYPES:
        raise ValueError(
            f"Env type {env_type!r} is evaluator-owned and has no local make_env builder."
        )

    return _make_lerobot_env(
        cast(Any, cfg_obj), n_envs=n_envs, use_async_envs=use_async_envs
    )


def _make_lerobot_env_config(env_type: str, **kwargs: Any) -> Any:
    from lerobot.envs.factory import make_env_config as lerobot_make_env_config

    return lerobot_make_env_config(env_type, **kwargs)


def _make_lerobot_env(cfg_obj: Any, *, n_envs: int, use_async_envs: bool) -> Any:
    from lerobot.envs.factory import make_env as lerobot_make_env

    return lerobot_make_env(
        cfg_obj,
        n_envs=n_envs,
        use_async_envs=use_async_envs,
    )


def flatten_envs(envs: dict[str, dict[int, Any]]) -> dict[str, Any]:
    """Flatten ``{suite: {task_id: env}}`` into ``{suite/task_id: env}``."""
    flat: dict[str, Any] = {}
    for suite, suite_envs in envs.items():
        for task_id, env in suite_envs.items():
            flat[f"{suite}/{task_id}"] = env
    return flat


def list_tasks(
    env_cfg: DictConfig | dict[str, Any],
    *,
    debug_verbose: bool = False,
) -> list[tuple[str, int]]:
    """List (suite, task_id) pairs without instantiating envs."""
    raw_cfg = _to_plain_dict(env_cfg)
    cfg_obj = build_env_config(env_cfg)
    env_type = _env_type_from_cfg(cfg_obj)
    lister = _TASK_LISTER_REGISTRY.get(env_type)
    if lister is None:
        return [(env_type, 0)]
    return _load_callable(lister)(raw_cfg, cfg_obj, debug_verbose)


def make_eval_async_pool(
    env_cfg: DictConfig | dict[str, Any],
    *,
    tasks: list[tuple[str, int]],
    n_envs: int,
    debug_verbose: bool = False,
) -> EvalPoolHandle:
    """Create one persistent async eval pool for the configured env type.

    This is backend-agnostic; env-specific details come from registered builders.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}.")
    if len(tasks) < 1:
        raise ValueError("tasks must be non-empty for make_eval_async_pool().")

    cfg_obj = build_env_config(env_cfg)
    env_type = _env_type_from_cfg(cfg_obj)
    builder = _EVAL_POOL_BUILDER_REGISTRY.get(env_type)
    if builder is None:
        raise ValueError(
            f"No persistent eval pool builder registered for env type {env_type!r}. "
            f"Available: {', '.join(available_eval_pool_env_types()) or '(none)'}"
        )
    return _load_callable(builder)(cfg_obj, tasks, n_envs, debug_verbose)


def infer_eval_env_target(dataset_name: str) -> tuple[str, str]:
    """Infer `(env.type, env.task)` from dataset naming conventions and env hooks."""
    for env_type in sorted(available_env_types(), key=len, reverse=True):
        if dataset_name == env_type or dataset_name.startswith(f"{env_type}_"):
            inferer = _EVAL_TARGET_INFERER_REGISTRY.get(env_type)
            if inferer is not None:
                inferred = _load_callable(inferer)(dataset_name)
                if inferred is not None:
                    return inferred
            return env_type, dataset_name
    for env_type in sorted(_EVAL_TARGET_INFERER_REGISTRY):
        inferred = _load_callable(_EVAL_TARGET_INFERER_REGISTRY[env_type])(dataset_name)
        if inferred is not None:
            return inferred
    return "", ""


def _to_plain_dict(env_cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(env_cfg, DictConfig):
        result = OmegaConf.to_container(env_cfg, resolve=True)
    else:
        result = dict(env_cfg)
    if not isinstance(result, dict):
        raise TypeError(f"env_cfg must resolve to dict, got {type(result)!r}")
    return {str(key): value for key, value in cast(dict[Any, Any], result).items()}


def _get_env_config_class(env_type: str) -> type:
    key = _normalize_type_name(env_type)
    if key not in _ENV_CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown env type: {env_type!r}. Available: {', '.join(available_env_types())}"
        )
    module_path, class_name = _ENV_CONFIG_REGISTRY[key].split(":", 1)
    module = importlib.import_module(module_path)
    cfg_cls = getattr(module, class_name, None)
    if cfg_cls is None:
        raise ImportError(f"Could not import {class_name!r} from {module_path!r}.")
    return cfg_cls


def _normalize_type_name(name: str) -> str:
    return str(name).strip().lower()


def _env_type_from_cfg(cfg_obj: Any) -> str:
    env_type = getattr(cfg_obj, "type", None)
    if env_type is None:
        raise ValueError("Environment config is missing required field `type`.")
    return _normalize_type_name(str(env_type))


_DEFAULT_ENV_FAMILY_REGISTRARS: tuple[str, ...] = (
    "praxis_eval.envs.libero.registration:register_libero_env_family",
    "praxis_eval.envs.metaworld.registration:register_metaworld_env_family",
    "praxis_eval.envs.simpler.registration:register_simpler_env_family",
    "praxis_eval.envs.mshab.registration:register_mshab_env_family",
    "praxis_eval.envs.robocasa_gr1.registration:register_robocasa_gr1_env_family",
)


def _load_callable(callback: Callable[..., Any] | str) -> Callable[..., Any]:
    if callable(callback):
        return callback
    module_path, function_name = str(callback).split(":", 1)
    module = importlib.import_module(module_path)
    loaded = getattr(module, function_name, None)
    if not callable(loaded):
        raise ImportError(
            f"Could not import callable {function_name!r} from {module_path!r}."
        )
    return cast(Callable[..., Any], loaded)


def _register_default_env_families() -> None:
    for import_path in _DEFAULT_ENV_FAMILY_REGISTRARS:
        _load_callable(import_path)()
    register_env_config(
        "robocasa", "praxis_eval.envs.robocasa.config:RobocasaEnvConfig"
    )
    register_task_lister(
        "robocasa", "praxis_eval.envs.robocasa.tasks:list_robocasa_tasks"
    )
    register_eval_target_inferer(
        "robocasa",
        "praxis_eval.envs.robocasa.tasks:infer_robocasa_eval_target_from_dataset",
    )
    register_eval_pool_builder(
        "robocasa",
        "praxis_eval.envs.robocasa.eval:build_robocasa_eval_pool",
    )
    register_env_config(
        "robomimic",
        "praxis_eval.envs.robomimic.config:RobomimicEnvConfig",
    )
    register_task_lister(
        "robomimic",
        "praxis_eval.envs.robomimic.tasks:list_robomimic_tasks",
    )
    register_eval_target_inferer(
        "robomimic",
        "praxis_eval.envs.robomimic.tasks:infer_robomimic_eval_target_from_dataset",
    )
    register_eval_pool_builder(
        "robomimic",
        "praxis_eval.envs.robomimic.eval:build_robomimic_eval_pool",
    )


_register_default_env_families()
