# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Verify RoboCasa GR-1 with a short physical-unit hold-position rollout."""

from __future__ import annotations

import argparse
import time

import numpy as np

from praxis_eval.scripts._verify_common import (
    print_summary,
    resolve_output_dir,
    validate_completed_episodes,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="PnPCupToDrawerClose")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=10)
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
        help="Gaussian noise added to the reset joint-position action.",
    )
    parser.add_argument(
        "--disable-render",
        action="store_true",
        help="Disable EGL/offscreen rendering for physics-only verification.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, "robocasa_gr1")

    from praxis_eval.envs.robocasa_gr1.env import RobocasaGr1Env
    from praxis_eval.envs.robocasa_gr1.spec import GR1_ACTION_KEYS, flatten_gr1_action

    rng = np.random.default_rng(int(args.seed))
    env = RobocasaGr1Env(
        str(args.task),
        max_episode_steps=int(args.max_episode_steps),
        enable_render=not bool(args.disable_render),
    )
    start = time.time()
    success = False
    terminated = False
    truncated = False
    total_reward = 0.0
    step_count = 0
    try:
        obs, _ = env.reset(seed=int(args.seed))
        task_description = str(env.task_description)
        for step_idx in range(1, int(args.max_episode_steps) + 1):
            step_count = step_idx
            streams = {
                key: obs["robot_state"][key.replace("action.", "state.")].copy()
                for key in GR1_ACTION_KEYS
            }
            action = flatten_gr1_action(streams)
            if float(args.action_noise) > 0:
                action += rng.normal(
                    0.0, float(args.action_noise), size=action.shape
                ).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            success = bool(info.get("is_success", False))
            if terminated or truncated:
                break
    finally:
        env.close()

    finished_reason = (
        "success"
        if success
        else "terminated"
        if terminated
        else "truncated"
        if truncated
        else "horizon"
    )
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
        "successes": [success],
        "task": str(args.task),
        "task_description": task_description,
        "verify_name": "verify_robocasa_gr1",
    }
    results_path = write_json(output_dir / "results.json", payload)
    validate_completed_episodes(verify_name="verify_robocasa_gr1", payload=payload)
    print_summary(
        verify_name="verify_robocasa_gr1",
        results_path=results_path,
        payload=payload,
    )


if __name__ == "__main__":
    main()
