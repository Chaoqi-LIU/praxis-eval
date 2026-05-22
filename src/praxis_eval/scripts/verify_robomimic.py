# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Verify that RoboMimic can roll out to completion with random actions."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from praxis_eval.scripts._verify_common import (
    print_summary,
    resolve_output_dir,
    validate_completed_episodes,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Lift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--video-resolution", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--camera-names", nargs="+", default=None)
    parser.add_argument("--state-ports", nargs="+", default=None)
    parser.add_argument("--video-camera", default="agentview")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument(
        "--disable-render",
        action="store_true",
        help="Skip offscreen rendering and verify physics stepping only.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _sample_action(env: Any, rng: np.random.Generator) -> np.ndarray:
    action_space = env.action_space
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    if low.shape != high.shape or low.shape != action_space.shape:
        raise RuntimeError(
            "RoboMimic action-space bounds do not match action-space shape: "
            f"low={low.shape}, high={high.shape}, shape={action_space.shape}."
        )
    if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
        return rng.uniform(low=low, high=high).astype(np.float32)

    return np.asarray(action_space.sample(), dtype=np.float32)


def _capture_render_frame(
    env: Any,
    *,
    output_dir: Path,
    expected_resolution: int,
) -> dict[str, object]:
    frame = env.render()
    expected_shape = (expected_resolution, expected_resolution, 3)
    if frame.shape != expected_shape:
        raise RuntimeError(
            "RoboMimic render returned an unexpected frame shape: "
            f"{frame.shape}; expected {expected_shape}."
        )
    if frame.dtype != np.uint8:
        raise RuntimeError(
            f"RoboMimic render returned dtype {frame.dtype}; expected uint8."
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

    output_dir = resolve_output_dir(args.output_dir, "robomimic")
    rng = np.random.default_rng(int(args.seed))
    enable_render = not bool(args.disable_render)
    from praxis_eval.envs.robomimic.env import RobomimicEnv

    env = RobomimicEnv(
        task_name=str(args.task),
        image_size=int(args.image_size),
        seed=int(args.seed),
        camera_names=list(args.camera_names) if args.camera_names is not None else None,
        state_ports=list(args.state_ports) if args.state_ports is not None else None,
        video_camera=str(args.video_camera),
        video_resolution=int(args.video_resolution),
        max_episode_steps=int(args.max_episode_steps),
        enable_render=enable_render,
        robot=str(args.robot),
    )

    start = time.time()
    success = False
    done = False
    truncated = False
    render_payload: dict[str, object] = {}
    total_reward = 0.0
    step_count = 0
    try:
        env.reset(seed=int(args.seed))
        if enable_render:
            render_payload = _capture_render_frame(
                env,
                output_dir=output_dir,
                expected_resolution=int(args.video_resolution),
            )
        for step_idx in range(1, int(args.max_episode_steps) + 1):
            step_count = step_idx
            action = _sample_action(env, rng)
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
        "camera_names": list(env.camera_names),
        "elapsed_s": float(time.time() - start),
        "episode_lengths": [int(step_count)],
        "enable_render": bool(enable_render),
        "finished_reason": finished_reason,
        "max_episode_steps": int(args.max_episode_steps),
        "n_episodes": 1.0,
        "state_ports": list(env.state_ports),
        "success_rate": float(success),
        "successes": [bool(success)],
        "task": str(args.task),
        "task_description": str(env.task_description),
        "verify_name": "verify_robomimic",
    }
    payload.update(render_payload)
    results_path = write_json(output_dir / "results.json", payload)
    validate_completed_episodes(verify_name="verify_robomimic", payload=payload)
    print_summary(
        verify_name="verify_robomimic",
        results_path=results_path,
        payload=payload,
    )


if __name__ == "__main__":
    main()
