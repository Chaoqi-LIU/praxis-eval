# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Subprocess-backed SimplerEnv evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from praxis_eval.envs.subprocess_runtime import (
    SubprocessTaskResult,
    append_policy_kwargs_json,
    append_video_recording_args,
    apply_task_id_filter,
    package_root_for_python,
    parse_host_port,
    raw_task_selectors,
    resolve_required_python_bin,
    run_subprocess_eval_flow,
    single_env_recording_context,
    weighted_metric,
    without_recording_context,
)
from praxis_eval.managed_paths import managed_asset_dir

if TYPE_CHECKING:
    from praxis_eval.envs.eval_registry import EvalDriverContext


@dataclass
class SimplerEnvConfig:
    """Configuration for SIMPLER evaluation."""

    processor_factory: ClassVar[str] = "identity"

    type: str = "simpler"
    task: str = "bridge"
    tasks: list[str] | None = None
    task_ids: list[int] | None = None
    python_bin: str | None = None
    policy_setup: str = "widowx_bridge"
    action_scale: float = 1.0
    primary_image_key: str = "observation.images.image"
    state_key: str = "observation.state"
    ms_asset_dir: str | None = None
    shader: str = "default"


@dataclass(frozen=True)
class SimplerTaskSpec:
    alias: str
    env_id: str
    task_id: int
    group: str = "bridge"
    policy_setup: str = "widowx_bridge"


_SIMPLER_BRIDGE_TASKS: tuple[SimplerTaskSpec, ...] = (
    SimplerTaskSpec(
        alias="widowx_carrot_on_plate",
        env_id="PutCarrotOnPlateInScene-v1",
        task_id=0,
    ),
    SimplerTaskSpec(
        alias="widowx_spoon_on_towel",
        env_id="PutSpoonOnTableClothInScene-v1",
        task_id=1,
    ),
    SimplerTaskSpec(
        alias="widowx_stack_cube",
        env_id="StackGreenCubeOnYellowCubeBakedTexInScene-v1",
        task_id=2,
    ),
    SimplerTaskSpec(
        alias="widowx_put_eggplant_in_basket",
        env_id="PutEggplantInBasketScene-v1",
        task_id=3,
    ),
)
_SIMPLER_GROUPS: dict[str, tuple[SimplerTaskSpec, ...]] = {
    "bridge": _SIMPLER_BRIDGE_TASKS,
    "bridge_mt4": _SIMPLER_BRIDGE_TASKS,
}
_SIMPLER_TASKS_BY_ALIAS = {spec.alias: spec for spec in _SIMPLER_BRIDGE_TASKS}
_SIMPLER_TASKS_BY_ENV_ID = {spec.env_id.lower(): spec for spec in _SIMPLER_BRIDGE_TASKS}


def run_simpler_eval(
    raw_cfg: dict[str, Any],
    cfg_obj: Any,
    context: EvalDriverContext,
) -> dict[str, Any]:
    """Run the SIMPLER eval flow and aggregate results."""
    task_specs = resolve_simpler_task_specs(raw_cfg, cfg_obj)
    return run_subprocess_eval_flow(
        display_name="SIMPLER",
        heartbeat_prefix="simpler",
        output_subdir="simpler",
        task_specs=task_specs,
        cfg_obj=cfg_obj,
        context=context,
        build_command=build_simpler_eval_command,
        cwd=lambda: resolve_simpler_runtime_root(
            python_bin=getattr(cfg_obj, "python_bin", None)
        ),
        summarize_runs=summarize_simpler_runs,
        metrics_context_for_task=without_recording_context,
        recording_context_for_task=single_env_recording_context,
        recording_note_for_task=simpler_recording_note_for_task,
    )


def list_simpler_tasks(
    raw_cfg: dict[str, Any],
    cfg_obj: Any,
    debug_verbose: bool = False,
) -> list[tuple[str, int]]:
    """List ``(task_group, task_id)`` pairs for the configured SIMPLER target."""
    del debug_verbose
    return [
        (spec.group, spec.task_id)
        for spec in resolve_simpler_task_specs(raw_cfg, cfg_obj)
    ]


def resolve_simpler_task_specs(
    raw_cfg: dict[str, Any],
    cfg_obj: Any,
) -> list[SimplerTaskSpec]:
    """Resolve configured SIMPLER task selectors to concrete task specs."""
    selectors = raw_task_selectors(
        raw_cfg,
        cfg_obj,
        default="bridge",
        label="SIMPLER",
    )
    ordered_specs: list[SimplerTaskSpec] = []
    seen_aliases: set[str] = set()
    for selector in selectors:
        for spec in expand_simpler_selector(selector):
            if spec.alias in seen_aliases:
                continue
            seen_aliases.add(spec.alias)
            ordered_specs.append(spec)

    ordered_specs = apply_task_id_filter(
        ordered_specs,
        task_ids=getattr(cfg_obj, "task_ids", None),
        label="Simpler",
    )
    if not ordered_specs:
        raise ValueError("Resolved zero SIMPLER tasks from the current env config.")
    return ordered_specs


def infer_simpler_eval_target_from_dataset(
    dataset_name: str,
) -> tuple[str, str] | None:
    """Infer SIMPLER eval target from known Bridge-family datasets."""
    normalized = str(dataset_name).strip().lower()
    if normalized == "bridge_single_view":
        return "simpler", "bridge"
    return None


def resolve_simpler_runtime_root(*, python_bin: str | None) -> Path:
    """Resolve the SimplerEnv source root inside the configured runtime."""
    env_python = resolve_simpler_python_bin(python_bin=python_bin)
    return package_root_for_python(env_python, "simpler_env").parent


def default_simpler_ms_asset_dir() -> Path:
    """Return the managed ManiSkill asset dir used by SIMPLER setup."""
    return managed_asset_dir("simpler")


def resolve_simpler_python_bin(*, python_bin: str | None) -> Path:
    """Resolve the Python interpreter for the external SIMPLER runtime."""
    return resolve_required_python_bin(
        python_bin=python_bin,
        display_name="SIMPLER",
        runtime_name="simpler-praxis interpreter",
        setup_command="python -m praxis_eval.scripts.setup_simpler",
    )


def build_simpler_eval_command(
    *,
    cfg_obj: SimplerEnvConfig,
    task_spec: SimplerTaskSpec,
    context: EvalDriverContext,
    server_address: str,
    metrics_output_path: Path,
    record_dir: Path,
) -> list[str]:
    """Build one external SIMPLER evaluation command."""
    host, port = parse_host_port(server_address)
    python_bin = resolve_simpler_python_bin(
        python_bin=getattr(cfg_obj, "python_bin", None)
    )
    ms_asset_dir = (
        Path(str(getattr(cfg_obj, "ms_asset_dir", None))).expanduser().resolve()
        if getattr(cfg_obj, "ms_asset_dir", None)
        else default_simpler_ms_asset_dir()
    )
    command = [
        "env",
        f"MS_ASSET_DIR={ms_asset_dir}",
        str(python_bin),
        "-u",
        "-m",
        "simpler_env.real2sim_eval_maniskill3",
        "--model",
        "praxis-remote",
        "--env-id",
        task_spec.env_id,
        "--seed",
        str(int(context.seed)),
        "--num-episodes",
        str(int(context.num_eval_per_task)),
        "--num-envs",
        str(int(context.num_parallel_env)),
        "--record-dir",
        str(record_dir.resolve()),
        "--praxis-host",
        host,
        "--praxis-port",
        str(port),
        "--praxis-policy-setup",
        str(
            getattr(cfg_obj, "policy_setup", task_spec.policy_setup)
            or task_spec.policy_setup
        ),
        "--praxis-action-scale",
        str(float(getattr(cfg_obj, "action_scale", 1.0))),
        "--praxis-primary-image-key",
        str(getattr(cfg_obj, "primary_image_key", "observation.images.image")),
        "--praxis-state-key",
        str(getattr(cfg_obj, "state_key", "observation.state")),
        "--shader",
        str(getattr(cfg_obj, "shader", "default")),
        "--metrics-output-path",
        str(metrics_output_path.resolve()),
    ]
    append_policy_kwargs_json(command, context.eval_policy_kwargs)
    append_video_recording_args(
        command,
        max_videos=int(context.eval_record_episodes_per_task),
    )
    return command


def simpler_recording_note_for_task(task_spec: SimplerTaskSpec) -> str:
    return (
        "SIMPLER note: recording videos with a dedicated single-env pass "
        f"for {task_spec.alias} to avoid multi-env video capture stalls."
    )


def summarize_simpler_runs(runs: list[SubprocessTaskResult]) -> dict[str, Any]:
    if not runs:
        raise ValueError("Cannot summarize zero SIMPLER task runs.")

    per_task: dict[str, dict[str, Any]] = {}
    per_group_raw: dict[str, list[SubprocessTaskResult]] = {}
    total_eval_s = 0.0
    total_episodes = 0.0
    all_video_paths: list[str] = []

    for run in runs:
        task_key = f"{run.spec.group}/{run.spec.task_id}"
        n_episodes = float(run.metrics["n_episodes"])
        per_task[task_key] = {
            "success_rate": float(run.metrics["success_rate"]),
            "avg_episode_length": float(run.metrics["avg_episode_length"]),
            "avg_reward": float(run.metrics["avg_reward"]),
            "avg_sum_reward": float(run.metrics["avg_reward"]),
            "avg_max_reward": float(run.metrics["avg_max_reward"]),
            "n_episodes": n_episodes,
            "task_alias": run.spec.alias,
            "env_id": run.spec.env_id,
            "task_description": str(
                run.metrics.get("task_description", run.spec.alias)
            ),
            "metrics_path": str(run.metrics_path),
            "log_path": str(run.log_path),
            "video_paths": list(run.video_paths),
        }
        task_descriptions = run.metrics.get("task_descriptions")
        if isinstance(task_descriptions, list):
            per_task[task_key]["task_descriptions"] = task_descriptions
        per_group_raw.setdefault(run.spec.group, []).append(run)
        total_eval_s += float(run.metrics.get("eval_s", 0.0))
        total_episodes += n_episodes
        all_video_paths.extend(run.video_paths)

    per_group: dict[str, dict[str, float]] = {}
    weighted_success = 0.0
    weighted_length = 0.0
    weighted_reward = 0.0
    weighted_max_reward = 0.0
    for group, group_runs in per_group_raw.items():
        group_episodes = sum(float(run.metrics["n_episodes"]) for run in group_runs)
        if group_episodes <= 0:
            continue
        group_success = weighted_metric(group_runs, "success_rate")
        group_length = weighted_metric(group_runs, "avg_episode_length")
        group_reward = weighted_metric(group_runs, "avg_reward")
        group_max_reward = weighted_metric(group_runs, "avg_max_reward")
        per_group[group] = {
            "success_rate": group_success,
            "avg_episode_length": group_length,
            "avg_reward": group_reward,
            "avg_sum_reward": group_reward,
            "avg_max_reward": group_max_reward,
            "n_episodes": group_episodes,
        }
        weighted_success += group_success * group_episodes
        weighted_length += group_length * group_episodes
        weighted_reward += group_reward * group_episodes
        weighted_max_reward += group_max_reward * group_episodes

    overall = {
        "success_rate": weighted_success / total_episodes if total_episodes else 0.0,
        "avg_episode_length": weighted_length / total_episodes
        if total_episodes
        else 0.0,
        "avg_reward": weighted_reward / total_episodes if total_episodes else 0.0,
        "avg_sum_reward": weighted_reward / total_episodes if total_episodes else 0.0,
        "avg_max_reward": weighted_max_reward / total_episodes
        if total_episodes
        else 0.0,
        "n_episodes": total_episodes,
        "eval_s": total_eval_s,
        "eval_ep_s": total_eval_s / max(1.0, total_episodes),
        "video_paths": all_video_paths,
    }
    return {
        "overall": overall,
        "per_group": per_group,
        "per_task": per_task,
    }


def expand_simpler_selector(selector: str) -> list[SimplerTaskSpec]:
    normalized = selector.strip()
    group_key = normalized.lower()
    if group_key in _SIMPLER_GROUPS:
        return list(_SIMPLER_GROUPS[group_key])
    alias = _SIMPLER_TASKS_BY_ALIAS.get(normalized)
    if alias is not None:
        return [alias]
    env_match = _SIMPLER_TASKS_BY_ENV_ID.get(group_key)
    if env_match is not None:
        return [env_match]
    available = ", ".join(sorted(list(_SIMPLER_GROUPS) + list(_SIMPLER_TASKS_BY_ALIAS)))
    raise ValueError(
        f"Unknown SIMPLER task selector {selector!r}. Available selectors: {available}"
    )
