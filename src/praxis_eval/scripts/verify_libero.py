# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Verify that LIBERO can roll out to completion with a random policy."""

from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import numpy as np

from praxis_eval.scripts._verify_common import (
    print_summary,
    resolve_output_dir,
    validate_completed_episodes,
    working_root,
    write_json,
)

TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
_NOOP_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument("--action-low", type=float, default=-0.25)
    parser.add_argument("--action-high", type=float, default=0.25)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _get_suite(name: str):
    benchmark, _get_libero_path, _offscreen_render_env = _get_libero_symbols()
    suite_factories = benchmark.get_benchmark_dict()
    if name not in suite_factories:
        raise ValueError(
            f"Unknown LIBERO suite {name!r}. Available: {sorted(suite_factories)}"
        )
    return suite_factories[name]()


def _ensure_libero_config() -> Path:
    config_root = working_root() / ".tmp" / "libero_config"
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_root))
    config_file = config_root / "config.yaml"
    if config_file.exists():
        return config_file

    spec = importlib.util.find_spec("libero.libero")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            "Could not locate the installed `libero.libero` package."
        )

    benchmark_root = Path(spec.origin).resolve().parent
    config_root.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {benchmark_root.as_posix()}",
                f"bddl_files: {(benchmark_root / 'bddl_files').as_posix()}",
                f"init_states: {(benchmark_root / 'init_files').as_posix()}",
                f"datasets: {(benchmark_root.parent / 'datasets').as_posix()}",
                f"assets: {(benchmark_root / 'assets').as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_file


def _get_libero_symbols():
    _ensure_libero_config()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, get_libero_path, OffScreenRenderEnv


def _get_init_states(*, suite, task_id: int, get_libero_path) -> np.ndarray:
    import torch

    task = suite.tasks[task_id]
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    return torch.load(init_states_path, weights_only=False)  # nosec B614


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, "libero")

    suite = _get_suite(str(args.task))
    _benchmark, get_libero_path, offscreen_render_env = _get_libero_symbols()
    task_id = int(args.task_id)
    task = suite.get_task(task_id)
    max_episode_steps = (
        int(args.max_episode_steps)
        if args.max_episode_steps is not None
        else TASK_SUITE_MAX_STEPS.get(str(args.task), 500)
    )
    rng = np.random.default_rng(int(args.seed))
    env = offscreen_render_env(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=int(args.image_size),
        camera_widths=int(args.image_size),
    )

    start = time.time()
    success = False
    done = False
    total_reward = 0.0
    step_count = 0
    try:
        env.seed(int(args.seed))
        env.reset()
        init_states = _get_init_states(
            suite=suite,
            task_id=task_id,
            get_libero_path=get_libero_path,
        )
        env.set_init_state(init_states[int(args.init_state_index) % len(init_states)])
        for _ in range(int(args.num_steps_wait)):
            env.step(_NOOP_ACTION)
        for robot in env.robots:
            controller = getattr(robot, "controller", None)
            if controller is not None and hasattr(controller, "use_delta"):
                controller.use_delta = True

        for step_idx in range(1, max_episode_steps + 1):
            step_count = step_idx
            action = rng.uniform(
                low=float(args.action_low),
                high=float(args.action_high),
                size=(7,),
            ).astype(np.float32)
            _obs, reward, done, _info = env.step(action)
            total_reward += float(reward)
            success = bool(env.check_success())
            if done or success:
                break
    finally:
        env.close()

    finished_reason = "success" if success else "done" if done else "horizon"
    payload = {
        "avg_episode_length": float(step_count),
        "avg_reward": float(total_reward),
        "avg_sum_reward": float(total_reward),
        "elapsed_s": float(time.time() - start),
        "episode_lengths": [int(step_count)],
        "finished_reason": finished_reason,
        "max_episode_steps": int(max_episode_steps),
        "n_episodes": 1.0,
        "success_rate": float(success),
        "successes": [bool(success)],
        "task": str(args.task),
        "task_description": str(task.language),
        "task_id": int(task_id),
        "task_name": str(task.name),
        "verify_name": "verify_libero",
    }
    results_path = write_json(output_dir / "results.json", payload)
    validate_completed_episodes(verify_name="verify_libero", payload=payload)
    print_summary(
        verify_name="verify_libero",
        results_path=results_path,
        payload=payload,
    )


if __name__ == "__main__":
    main()
