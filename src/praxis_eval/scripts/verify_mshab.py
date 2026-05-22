# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Verify that MS-HAB can roll out to completion with random actions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from praxis_eval.managed_paths import managed_asset_dir
from praxis_eval.scripts._verify_common import (
    env_or_default_path,
    load_json,
    print_summary,
    resolve_env_python_bin,
    resolve_output_dir,
    run_command,
    validate_completed_episodes,
)

_DEFAULT_TARGETS = {
    "pick": "all",
    "place": "all",
    "open": "fridge",
    "close": "fridge",
}


def default_ms_asset_dir() -> Path:
    return env_or_default_path(
        "MS_ASSET_DIR",
        managed_asset_dir("mshab"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="set_table")
    parser.add_argument(
        "--subtask",
        default="pick",
        choices=("pick", "place", "open", "close"),
    )
    parser.add_argument("--target", default=None)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--env-name", default="mshab-praxis")
    parser.add_argument("--env-python-bin", default=None)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ms-asset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metrics-output-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--runtime-mode", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def rearrange_root(ms_asset_dir: Path) -> Path:
    return (
        ms_asset_dir / "data" / "scene_datasets" / "replica_cad_dataset" / "rearrange"
    )


def task_plan_path(
    *, root: Path, task: str, subtask: str, split: str, target: str
) -> Path:
    return root / "task_plans" / task / subtask / split / f"{target}.json"


def spawn_data_path(*, root: Path, task: str, subtask: str, split: str) -> Path:
    return root / "spawn_data" / task / subtask / split / "spawn_data.pt"


def launch_runtime(args: argparse.Namespace) -> None:
    env_python_bin = resolve_env_python_bin(args.env_python_bin, str(args.env_name))
    output_dir = resolve_output_dir(args.output_dir, "mshab")
    output_dir.mkdir(parents=True, exist_ok=True)

    subtask = str(args.subtask)
    target = str(args.target or _DEFAULT_TARGETS[subtask])
    ms_asset_dir = (
        Path(args.ms_asset_dir).expanduser().resolve()
        if args.ms_asset_dir is not None
        else default_ms_asset_dir()
    )
    results_path = output_dir / "results.json"
    command = [
        env_python_bin,
        "-u",
        "-m",
        "praxis_eval.scripts.verify_mshab",
        "--task",
        str(args.task),
        "--subtask",
        subtask,
        "--target",
        target,
        "--split",
        str(args.split),
        "--num-episodes",
        str(int(args.num_episodes)),
        "--num-envs",
        str(int(args.num_envs)),
        "--seed",
        str(int(args.seed)),
        "--ms-asset-dir",
        ms_asset_dir,
        "--runtime-mode",
        "--metrics-output-path",
        results_path,
    ]
    run_command(command)
    payload = load_json(results_path)
    validate_completed_episodes(verify_name="verify_mshab", payload=payload)
    print_summary(
        verify_name="verify_mshab",
        results_path=results_path,
        payload=payload,
    )


def run_runtime(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from mshab.runtime_bootstrap import load_env_factory

    def to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def extend_done_values(dest: list[Any], value: Any, done_mask: np.ndarray) -> None:
        dest.extend(to_numpy(value)[done_mask].tolist())

    metrics_output_path_value = args.metrics_output_path
    if metrics_output_path_value is None:
        raise ValueError("--metrics-output-path is required in --runtime-mode.")

    ms_asset_dir, EnvConfig, make_env = load_env_factory(args.ms_asset_dir)
    root = rearrange_root(ms_asset_dir)
    subtask = str(args.subtask)
    target = str(args.target or _DEFAULT_TARGETS[subtask])
    plan_fp = task_plan_path(
        root=root,
        task=str(args.task),
        subtask=subtask,
        split=str(args.split),
        target=target,
    )
    spawn_fp = spawn_data_path(
        root=root,
        task=str(args.task),
        subtask=subtask,
        split=str(args.split),
    )
    if not plan_fp.exists():
        raise FileNotFoundError(f"Missing task plan file: {plan_fp}")
    if not spawn_fp.exists():
        raise FileNotFoundError(f"Missing spawn data file: {spawn_fp}")

    env_cfg = EnvConfig(
        env_id=f"{subtask.capitalize()}SubtaskTrain-v0",
        num_envs=int(args.num_envs),
        max_episode_steps=200,
        obs_mode="depth",
        frame_stack=3,
        cat_state=True,
        cat_pixels=False,
        record_video=False,
        task_plan_fp=str(plan_fp),
        spawn_data_fp=str(spawn_fp),
        env_kwargs={
            "require_build_configs_repeated_equally_across_envs": False,
            "add_event_tracker_info": True,
        },
    )

    envs = None
    start = time.time()
    lengths: list[int] = []
    sum_rewards: list[float] = []
    success_once: list[bool] = []
    success_at_end: list[bool] = []
    try:
        envs = make_env(env_cfg)
        device = envs.unwrapped.device
        envs.reset(seed=int(args.seed), options={"reconfigure": True})
        while len(lengths) < int(args.num_episodes):
            action = envs.action_space.sample()
            if torch.is_tensor(action):
                action_tensor = action.to(device=device)
            else:
                action_tensor = torch.as_tensor(action, device=device)
            _obs, _reward, _terminated, _truncated, infos = envs.step(action_tensor)
            done_mask = to_numpy(
                infos.get("_episode", np.zeros((int(args.num_envs),), dtype=bool))
            ).astype(bool)
            if np.any(done_mask):
                episode_info = infos["episode"]
                extend_done_values(sum_rewards, episode_info["r"], done_mask)
                extend_done_values(lengths, episode_info["l"], done_mask)
                extend_done_values(success_once, episode_info["s_o"], done_mask)
                extend_done_values(success_at_end, episode_info["s_e"], done_mask)
    finally:
        if envs is not None:
            envs.close()

    limit = int(args.num_episodes)
    payload = {
        "avg_episode_length": float(np.mean(lengths[:limit])) if lengths else 0.0,
        "avg_reward": float(np.mean(sum_rewards[:limit])) if sum_rewards else 0.0,
        "avg_sum_reward": float(np.mean(sum_rewards[:limit])) if sum_rewards else 0.0,
        "elapsed_s": float(time.time() - start),
        "lengths": [int(value) for value in lengths[:limit]],
        "max_episode_steps": 200,
        "n_episodes": float(min(len(lengths), limit)),
        "spawn_data_fp": str(spawn_fp),
        "subtask": subtask,
        "success_at_end": [bool(value) for value in success_at_end[:limit]],
        "success_at_end_rate": (
            float(np.mean(success_at_end[:limit])) if success_at_end else 0.0
        ),
        "success_once": [bool(value) for value in success_once[:limit]],
        "success_once_rate": (
            float(np.mean(success_once[:limit])) if success_once else 0.0
        ),
        "success_rate": (
            float(np.mean(success_at_end[:limit])) if success_at_end else 0.0
        ),
        "sum_rewards": [float(value) for value in sum_rewards[:limit]],
        "target": target,
        "task": str(args.task),
        "task_plan_fp": str(plan_fp),
        "verify_name": "verify_mshab",
    }
    metrics_output_path = Path(metrics_output_path_value).expanduser().resolve()
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.runtime_mode:
        run_runtime(args)
        return
    launch_runtime(args)


if __name__ == "__main__":
    main()
