# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Verify that RoboCasa can roll out to completion with random actions."""

from __future__ import annotations

import argparse
import time
from typing import Literal, cast

import numpy as np

from praxis_eval.scripts._verify_common import (
    print_summary,
    resolve_output_dir,
    validate_completed_episodes,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="CloseToasterOvenDoor")
    parser.add_argument("--split", default="all", choices=("all", "pretrain", "target"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--action-low", type=float, default=-0.25)
    parser.add_argument("--action-high", type=float, default=0.25)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, "robocasa")

    rng = np.random.default_rng(int(args.seed))
    split = cast(Literal["all", "pretrain", "target"], str(args.split))
    from praxis_eval.envs.robocasa.env import RobocasaEnv

    env = RobocasaEnv(
        task_name=str(args.task),
        split=split,
        image_size=int(args.image_size),
        seed=int(args.seed),
        max_episode_steps=int(args.max_episode_steps),
    )
    start = time.time()
    success = False
    done = False
    truncated = False
    total_reward = 0.0
    step_count = 0
    task_description = str(args.task)
    try:
        env.reset(seed=int(args.seed))
        task_description = str(env.task_description)
        raw_action_shape = env.action_space.shape
        if raw_action_shape is None:
            raise RuntimeError("RoboCasa action space has no shape.")
        action_shape = tuple(int(dim) for dim in raw_action_shape)
        for step_idx in range(1, int(args.max_episode_steps) + 1):
            step_count = step_idx
            action = rng.uniform(
                low=float(args.action_low),
                high=float(args.action_high),
                size=action_shape,
            ).astype(np.float32)
            _obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            success = bool(info.get("is_success", False))
            if done or truncated:
                break
    finally:
        env.close()

    if success:
        finished_reason = "success"
    elif done:
        finished_reason = "done"
    elif truncated:
        finished_reason = "truncated"
    else:
        finished_reason = "horizon"
    payload = {
        "avg_episode_length": float(step_count),
        "avg_reward": float(total_reward),
        "avg_sum_reward": float(total_reward),
        "elapsed_s": float(time.time() - start),
        "episode_lengths": [int(step_count)],
        "finished_reason": finished_reason,
        "max_episode_steps": int(args.max_episode_steps),
        "n_episodes": 1.0,
        "success_rate": float(success),
        "successes": [bool(success)],
        "task": str(args.task),
        "task_description": task_description,
        "verify_name": "verify_robocasa",
    }
    results_path = write_json(output_dir / "results.json", payload)
    validate_completed_episodes(verify_name="verify_robocasa", payload=payload)
    print_summary(
        verify_name="verify_robocasa",
        results_path=results_path,
        payload=payload,
    )


if __name__ == "__main__":
    main()
