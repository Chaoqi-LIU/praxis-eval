# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Verify that MetaWorld can roll out to completion with random actions."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from praxis_eval.scripts._verify_common import (
    print_summary,
    resolve_output_dir,
    validate_completed_episodes,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="reach-v3")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--camera-name", default="corner2")
    parser.add_argument("--obs-type", default="pixels_agent_pos")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _sample_action(env, rng: np.random.Generator) -> np.ndarray:
    action_space = env.action_space
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    if low.shape != high.shape or low.shape != action_space.shape:
        raise RuntimeError(
            "MetaWorld action-space bounds do not match action-space shape: "
            f"low={low.shape}, high={high.shape}, shape={action_space.shape}."
        )
    return rng.uniform(low=low, high=high).astype(np.float32)


def _capture_render_frame(
    env,
    *,
    output_dir: Path,
    expected_resolution: int,
) -> dict[str, object]:
    frame = env.render()
    expected_shape = (expected_resolution, expected_resolution, 3)
    if frame.shape != expected_shape:
        raise RuntimeError(
            "MetaWorld render returned an unexpected frame shape: "
            f"{frame.shape}; expected {expected_shape}."
        )
    if frame.dtype != np.uint8:
        raise RuntimeError(
            f"MetaWorld render returned dtype {frame.dtype}; expected uint8."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "first_render.npy"
    np.save(frame_path, frame)
    return {
        "render_frame_path": str(frame_path),
        "render_frame_shape": [int(dim) for dim in frame.shape],
        "render_frame_mean": float(np.mean(frame)),
        "render_frame_std": float(np.std(frame)),
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    output_dir = resolve_output_dir(args.output_dir, "metaworld")

    from praxis_eval.envs.metaworld.runtime import make_metaworld_env_fn
    from praxis_eval.envs.metaworld.tasks import resolve_task_name

    task_name = resolve_task_name(str(args.task), int(args.task_id))
    rng = np.random.default_rng(int(args.seed))
    env = make_metaworld_env_fn(
        task_name=task_name,
        camera_name=str(args.camera_name),
        obs_type=str(args.obs_type),
        observation_width=int(args.image_size),
        observation_height=int(args.image_size),
        visualization_width=int(args.image_size),
        visualization_height=int(args.image_size),
        episode_length=args.max_episode_steps,
    )()

    start = time.time()
    success = False
    terminated = False
    truncated = False
    total_reward = 0.0
    step_count = 0
    task_description = str(env.task_description)
    max_episode_steps = (
        int(args.max_episode_steps)
        if args.max_episode_steps is not None
        else int(env._max_episode_steps)
    )
    render_payload: dict[str, object] = {}
    try:
        env.reset(seed=int(args.seed))
        render_payload = _capture_render_frame(
            env,
            output_dir=output_dir,
            expected_resolution=int(args.image_size),
        )

        for step_idx in range(1, max_episode_steps + 1):
            step_count = step_idx
            action = _sample_action(env, rng)
            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            success = bool(info.get("is_success", False))
            if terminated or truncated:
                break
    finally:
        env.close()

    if success:
        finished_reason = "success"
    elif terminated:
        finished_reason = "terminated"
    elif truncated:
        finished_reason = "truncated"
    else:
        finished_reason = "horizon"

    payload = {
        "avg_episode_length": float(step_count),
        "avg_reward": float(total_reward),
        "avg_sum_reward": float(total_reward),
        "camera_name": str(args.camera_name),
        "elapsed_s": float(time.time() - start),
        "episode_lengths": [int(step_count)],
        "finished_reason": finished_reason,
        "max_episode_steps": int(max_episode_steps),
        "n_episodes": 1.0,
        "obs_type": str(args.obs_type),
        "success_rate": float(success),
        "successes": [bool(success)],
        "task": str(args.task),
        "task_description": task_description,
        "task_id": int(args.task_id),
        "task_name": task_name,
        "verify_name": "verify_metaworld",
    }
    payload.update(render_payload)
    results_path = write_json(output_dir / "results.json", payload)
    validate_completed_episodes(verify_name="verify_metaworld", payload=payload)
    print_summary(
        verify_name="verify_metaworld",
        results_path=results_path,
        payload=payload,
    )


if __name__ == "__main__":
    main()
