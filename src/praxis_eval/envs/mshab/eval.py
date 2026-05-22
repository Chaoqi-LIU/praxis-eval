# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Subprocess-backed MS-HAB evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

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
    sum_metric,
    weighted_metric,
    without_recording_context,
)
from praxis_eval.managed_paths import managed_asset_dir

if TYPE_CHECKING:
    from praxis_eval.envs.eval_registry import EvalDriverContext


@dataclass
class MshabEnvConfig:
    """Configuration for MS-HAB subtask evaluation."""

    processor_factory: ClassVar[str] = "identity"

    type: str = "mshab"
    task: str = "set_table"
    tasks: list[str] | None = None
    task_ids: list[int] | None = None
    python_bin: str | None = None
    split: Literal["train", "val"] = "train"
    ms_asset_dir: str | None = None
    obs_mode: Literal["rgb", "rgbd", "depth"] = "rgb"
    frame_stack: int = 1
    stationary_base: bool = False
    stationary_torso: bool = False
    stationary_head: bool = True
    debug_max_control_steps: int | None = None


@dataclass(frozen=True)
class MshabTaskSpec:
    alias: str
    subtask: str
    target: str
    task_id: int
    env_id: str
    policy_task: str
    task_description: str
    group: str = "set_table"


_MSHAB_SET_TABLE_TASKS: tuple[MshabTaskSpec, ...] = (
    MshabTaskSpec(
        alias="set_table_pick",
        subtask="pick",
        target="all",
        task_id=0,
        env_id="PickSubtaskTrain-v0",
        policy_task="set table: pick object",
        task_description="set table: pick object",
    ),
    MshabTaskSpec(
        alias="set_table_place",
        subtask="place",
        target="all",
        task_id=1,
        env_id="PlaceSubtaskTrain-v0",
        policy_task="set table: place object",
        task_description="set table: place object",
    ),
    MshabTaskSpec(
        alias="set_table_open_fridge",
        subtask="open",
        target="fridge",
        task_id=2,
        env_id="OpenSubtaskTrain-v0",
        policy_task="set table: open fridge",
        task_description="set table: open fridge",
    ),
    MshabTaskSpec(
        alias="set_table_open_kitchen_counter",
        subtask="open",
        target="kitchen_counter",
        task_id=3,
        env_id="OpenSubtaskTrain-v0",
        policy_task="set table: open drawer",
        task_description="set table: open drawer",
    ),
    MshabTaskSpec(
        alias="set_table_close_fridge",
        subtask="close",
        target="fridge",
        task_id=4,
        env_id="CloseSubtaskTrain-v0",
        policy_task="set table: close fridge",
        task_description="set table: close fridge",
    ),
    MshabTaskSpec(
        alias="set_table_close_kitchen_counter",
        subtask="close",
        target="kitchen_counter",
        task_id=5,
        env_id="CloseSubtaskTrain-v0",
        policy_task="set table: close drawer",
        task_description="set table: close drawer",
    ),
)
_MSHAB_GROUPS: dict[str, tuple[MshabTaskSpec, ...]] = {
    "set_table": _MSHAB_SET_TABLE_TASKS,
    "settable": _MSHAB_SET_TABLE_TASKS,
}
_MSHAB_TASKS_BY_ALIAS: dict[str, MshabTaskSpec] = {
    spec.alias: spec for spec in _MSHAB_SET_TABLE_TASKS
}
_MSHAB_TASKS_BY_SELECTOR: dict[str, MshabTaskSpec] = {
    "pick": _MSHAB_TASKS_BY_ALIAS["set_table_pick"],
    "place": _MSHAB_TASKS_BY_ALIAS["set_table_place"],
    "open_fridge": _MSHAB_TASKS_BY_ALIAS["set_table_open_fridge"],
    "open_kitchen_counter": _MSHAB_TASKS_BY_ALIAS["set_table_open_kitchen_counter"],
    "close_fridge": _MSHAB_TASKS_BY_ALIAS["set_table_close_fridge"],
    "close_kitchen_counter": _MSHAB_TASKS_BY_ALIAS["set_table_close_kitchen_counter"],
}
_MSHAB_CLEAN_PICK_PLACE_DATASETS: frozenset[str] = frozenset(
    {
        "mshab_settable",
        "mshab_set_table",
        "mshab_settable_clean",
        "mshab_set_table_clean",
    }
)
_MSHAB_FULL_SET_TABLE_DATASETS: frozenset[str] = frozenset(
    {
        "mshab_settable_full",
        "mshab_set_table_full",
        "mshab_settable_w_depth",
        "mshab_set_table_w_depth",
    }
)


def run_mshab_eval(
    raw_cfg: dict[str, Any],
    cfg_obj: Any,
    context: EvalDriverContext,
) -> dict[str, Any]:
    """Run the MS-HAB eval flow and aggregate results."""
    task_specs = resolve_mshab_task_specs(raw_cfg, cfg_obj)
    return run_subprocess_eval_flow(
        display_name="MS-HAB",
        heartbeat_prefix="mshab",
        output_subdir="mshab",
        task_specs=task_specs,
        cfg_obj=cfg_obj,
        context=context,
        build_command=build_mshab_eval_command,
        cwd=lambda: resolve_mshab_runtime_root(
            python_bin=getattr(cfg_obj, "python_bin", None)
        ),
        summarize_runs=summarize_mshab_runs,
        metrics_context_for_task=without_recording_context,
        recording_context_for_task=single_env_recording_context,
        recording_note_for_task=mshab_recording_note_for_task,
    )


def list_mshab_tasks(
    raw_cfg: dict[str, Any],
    cfg_obj: Any,
    debug_verbose: bool = False,
) -> list[tuple[str, int]]:
    """List ``(task_group, task_id)`` pairs for the configured MS-HAB target."""
    del debug_verbose
    return [
        (spec.group, spec.task_id)
        for spec in resolve_mshab_task_specs(raw_cfg, cfg_obj)
    ]


def resolve_mshab_task_specs(
    raw_cfg: dict[str, Any],
    cfg_obj: Any,
) -> list[MshabTaskSpec]:
    """Resolve configured MS-HAB task selectors to concrete task specs."""
    selectors = raw_task_selectors(
        raw_cfg,
        cfg_obj,
        default="set_table",
        label="MS-HAB",
    )
    ordered_specs: list[MshabTaskSpec] = []
    seen_aliases: set[str] = set()
    for selector in selectors:
        for spec in expand_mshab_selector(selector):
            if spec.alias in seen_aliases:
                continue
            seen_aliases.add(spec.alias)
            ordered_specs.append(spec)

    ordered_specs = apply_task_id_filter(
        ordered_specs,
        task_ids=getattr(cfg_obj, "task_ids", None),
        label="MS-HAB",
    )
    if not ordered_specs:
        raise ValueError("Resolved zero MS-HAB tasks from the current env config.")
    return ordered_specs


def infer_mshab_eval_target_from_dataset(
    dataset_name: str,
) -> tuple[str, str] | None:
    """Infer ``(env.type, env.task)`` from known MS-HAB dataset names."""
    normalized = str(dataset_name).strip().lower()
    if normalized in _MSHAB_CLEAN_PICK_PLACE_DATASETS:
        return "mshab", "pick,place"
    if normalized in _MSHAB_FULL_SET_TABLE_DATASETS:
        return "mshab", "set_table"
    return None


def resolve_mshab_runtime_root(*, python_bin: str | None) -> Path:
    """Resolve the MS-HAB source root inside the configured runtime."""
    env_python = resolve_mshab_python_bin(python_bin=python_bin)
    return package_root_for_python(env_python, "mshab").parent


def default_mshab_ms_asset_dir() -> Path:
    """Return the managed ManiSkill asset dir used by MS-HAB setup."""
    return managed_asset_dir("mshab")


def resolve_mshab_python_bin(*, python_bin: str | None) -> Path:
    """Resolve the Python interpreter for the external MS-HAB runtime."""
    return resolve_required_python_bin(
        python_bin=python_bin,
        display_name="MS-HAB",
        runtime_name="mshab-praxis interpreter",
        setup_command="python -m praxis_eval.scripts.setup_mshab",
    )


def build_mshab_eval_command(
    *,
    cfg_obj: MshabEnvConfig,
    task_spec: MshabTaskSpec,
    context: EvalDriverContext,
    server_address: str,
    metrics_output_path: Path,
    record_dir: Path,
) -> list[str]:
    """Build one external MS-HAB evaluation command."""
    host, port = parse_host_port(server_address)
    python_bin = resolve_mshab_python_bin(
        python_bin=getattr(cfg_obj, "python_bin", None)
    )
    ms_asset_dir = (
        Path(str(getattr(cfg_obj, "ms_asset_dir", None))).expanduser().resolve()
        if getattr(cfg_obj, "ms_asset_dir", None)
        else default_mshab_ms_asset_dir()
    )
    command = [
        "env",
        f"MS_ASSET_DIR={ms_asset_dir}",
        str(python_bin),
        "-u",
        "-m",
        "mshab.praxis_eval",
        "--task",
        task_spec.group,
        "--subtask",
        task_spec.subtask,
        "--target",
        task_spec.target,
        "--task-alias",
        task_spec.alias,
        "--policy-task",
        task_spec.policy_task,
        "--task-description",
        task_spec.task_description,
        "--split",
        str(getattr(cfg_obj, "split", "train")),
        "--ms-asset-dir",
        str(ms_asset_dir),
        "--num-episodes",
        str(int(context.num_eval_per_task)),
        "--num-envs",
        str(int(context.num_parallel_env)),
        "--seed",
        str(int(context.seed)),
        "--obs-mode",
        str(getattr(cfg_obj, "obs_mode", "rgb")),
        "--frame-stack",
        str(int(getattr(cfg_obj, "frame_stack", 1))),
        "--praxis-host",
        host,
        "--praxis-port",
        str(port),
        "--metrics-output-path",
        str(metrics_output_path.resolve()),
        "--record-dir",
        str(record_dir.resolve()),
    ]
    if bool(getattr(cfg_obj, "stationary_base", False)):
        command.append("--stationary-base")
    if bool(getattr(cfg_obj, "stationary_torso", False)):
        command.append("--stationary-torso")
    if bool(getattr(cfg_obj, "stationary_head", True)):
        command.append("--stationary-head")
    else:
        command.append("--no-stationary-head")
    append_policy_kwargs_json(command, context.eval_policy_kwargs)
    debug_max_control_steps = getattr(cfg_obj, "debug_max_control_steps", None)
    if debug_max_control_steps is not None:
        command.extend(["--debug-max-control-steps", str(int(debug_max_control_steps))])
    append_video_recording_args(
        command,
        max_videos=int(context.eval_record_episodes_per_task),
    )
    return command


def mshab_recording_note_for_task(task_spec: MshabTaskSpec) -> str:
    return (
        "MS-HAB note: recording videos with a dedicated single-env pass "
        f"for {task_spec.alias} to avoid giant tiled multi-env videos."
    )


def summarize_mshab_runs(runs: list[SubprocessTaskResult]) -> dict[str, Any]:
    if not runs:
        raise ValueError("Cannot summarize zero MS-HAB task runs.")

    per_task: dict[str, dict[str, Any]] = {}
    per_group_raw: dict[str, list[SubprocessTaskResult]] = {}
    total_eval_s = 0.0
    total_episodes = 0.0
    all_video_paths: list[str] = []

    for run in runs:
        task_key = f"{run.spec.group}/{run.spec.task_id}"
        n_episodes = float(run.metrics["n_episodes"])
        avg_sum_reward = float(run.metrics["avg_sum_reward"])
        per_task[task_key] = {
            "success_rate": float(run.metrics["success_once_rate"]),
            "success_once_rate": float(run.metrics["success_once_rate"]),
            "success_at_end_rate": float(run.metrics["success_at_end_rate"]),
            "avg_episode_length": float(run.metrics["avg_episode_length"]),
            "avg_reward": avg_sum_reward,
            "avg_sum_reward": avg_sum_reward,
            "avg_max_reward": 0.0,
            "avg_return_per_step": float(run.metrics["avg_return_per_step"]),
            "n_episodes": n_episodes,
            "task_alias": run.spec.alias,
            "env_id": run.spec.env_id,
            "task_description": run.spec.task_description,
            "metrics_path": str(run.metrics_path),
            "log_path": str(run.log_path),
            "video_paths": list(run.video_paths),
        }
        per_group_raw.setdefault(run.spec.group, []).append(run)
        total_eval_s += float(run.metrics.get("eval_s", 0.0))
        total_episodes += n_episodes
        all_video_paths.extend(run.video_paths)

    per_group: dict[str, dict[str, float]] = {}
    weighted_success_once = 0.0
    weighted_success_at_end = 0.0
    weighted_length = 0.0
    weighted_sum_reward = 0.0
    weighted_return_per_step = 0.0
    for group, group_runs in per_group_raw.items():
        group_episodes = sum_metric(group_runs, "n_episodes")
        if group_episodes <= 0:
            continue
        group_success_once = weighted_metric(group_runs, "success_once_rate")
        group_success_at_end = weighted_metric(group_runs, "success_at_end_rate")
        group_length = weighted_metric(group_runs, "avg_episode_length")
        group_sum_reward = weighted_metric(group_runs, "avg_sum_reward")
        group_return_per_step = weighted_metric(group_runs, "avg_return_per_step")
        per_group[group] = {
            "success_rate": group_success_once,
            "success_once_rate": group_success_once,
            "success_at_end_rate": group_success_at_end,
            "avg_episode_length": group_length,
            "avg_reward": group_sum_reward,
            "avg_sum_reward": group_sum_reward,
            "avg_max_reward": 0.0,
            "avg_return_per_step": group_return_per_step,
            "n_episodes": group_episodes,
        }
        weighted_success_once += group_success_once * group_episodes
        weighted_success_at_end += group_success_at_end * group_episodes
        weighted_length += group_length * group_episodes
        weighted_sum_reward += group_sum_reward * group_episodes
        weighted_return_per_step += group_return_per_step * group_episodes

    overall = {
        "success_rate": weighted_success_once / total_episodes
        if total_episodes
        else 0.0,
        "success_once_rate": weighted_success_once / total_episodes
        if total_episodes
        else 0.0,
        "success_at_end_rate": weighted_success_at_end / total_episodes
        if total_episodes
        else 0.0,
        "avg_episode_length": weighted_length / total_episodes
        if total_episodes
        else 0.0,
        "avg_reward": weighted_sum_reward / total_episodes if total_episodes else 0.0,
        "avg_sum_reward": weighted_sum_reward / total_episodes
        if total_episodes
        else 0.0,
        "avg_max_reward": 0.0,
        "avg_return_per_step": weighted_return_per_step / total_episodes
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


def expand_mshab_selector(selector: str) -> list[MshabTaskSpec]:
    normalized = selector.strip().lower()
    if normalized in _MSHAB_GROUPS:
        return list(_MSHAB_GROUPS[normalized])
    task_spec = _MSHAB_TASKS_BY_ALIAS.get(normalized) or _MSHAB_TASKS_BY_SELECTOR.get(
        normalized
    )
    if task_spec is not None:
        return [task_spec]
    available = ", ".join(sorted(list(_MSHAB_GROUPS) + list(_MSHAB_TASKS_BY_SELECTOR)))
    raise ValueError(
        f"Unknown MS-HAB task selector {selector!r}. Available selectors: {available}"
    )
