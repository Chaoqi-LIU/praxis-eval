from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from praxis_eval import (
    ActionSpec,
    EnvContract,
    EvalConfig,
    EvalResult,
    ObservationKey,
    register_driver,
)
from praxis_eval.types import Policy


class PointReachDriver:
    """Tiny in-memory driver used by the examples.

    The state is a two-dimensional point. Each action is a clipped delta in
    normalized action space. This is intentionally simple so examples can focus
    on the `praxis-eval` API rather than simulator setup.
    """

    @property
    def contract(self) -> EnvContract:
        return EnvContract(
            env_type="point_reach",
            observation_keys=(
                ObservationKey(
                    "task",
                    "str",
                    description="Natural-language task instruction.",
                ),
                ObservationKey(
                    "observation.state",
                    "float32",
                    shape=(2,),
                    description="Current xy point position.",
                ),
                ObservationKey(
                    "observation.goal",
                    "float32",
                    shape=(2,),
                    description="Target xy point position for this episode.",
                ),
                ObservationKey(
                    "metadata.step",
                    "int",
                    description="Zero-based step index within the episode.",
                ),
            ),
            action=ActionSpec(
                shape=(2,),
                dtype="float32",
                minimum=-1.0,
                maximum=1.0,
                convention="Normalized xy delta. The driver multiplies it by action_scale.",
                description="Move the point toward the target.",
            ),
            notes="Example-only driver. It has no simulator dependency.",
        )

    def evaluate(self, *, policy: Policy, config: EvalConfig) -> EvalResult:
        if config.num_eval_per_task < 1:
            raise ValueError("num_eval_per_task must be >= 1")

        task = config.task or "move the point to the target"
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(config.seed)
        target = _vector2(config.env_kwargs.get("target", (0.0, 0.0)), name="target")
        action_scale = float(config.env_kwargs.get("action_scale", 0.25))
        max_steps = int(config.env_kwargs.get("max_steps", 20))
        success_threshold = float(config.env_kwargs.get("success_threshold", 0.05))

        policy.reset()
        episodes: list[dict[str, Any]] = []
        for episode_index in range(config.num_eval_per_task):
            episode_id = f"{task}/{episode_index}"
            policy.reset(episode_ids=[episode_id])
            position = _initial_position(config, rng)
            trace: list[dict[str, Any]] = []
            success = False

            for step in range(max_steps):
                observation = {
                    "task": task,
                    "observation.state": position.astype(np.float32),
                    "observation.goal": target.astype(np.float32),
                    "metadata.step": step,
                }
                action_batch = policy.act(
                    [observation],
                    action_spec=self.contract.action,
                    policy_kwargs=config.policy_kwargs,
                    episode_ids=[episode_id],
                )
                action = self.contract.action.validate_batch(
                    action_batch,
                    batch_size=1,
                )[0]
                position = position + action.astype(np.float32) * action_scale
                distance = float(np.linalg.norm(target - position))
                trace.append(
                    {
                        "step": step,
                        "position": position.astype(float).tolist(),
                        "action": action.astype(float).tolist(),
                        "distance": distance,
                    }
                )
                if distance <= success_threshold:
                    success = True
                    break

            final_distance = float(np.linalg.norm(target - position))
            episodes.append(
                {
                    "episode_id": episode_id,
                    "success": success,
                    "episode_length": len(trace),
                    "final_distance": final_distance,
                    "trace": trace,
                }
            )

        success_rate = sum(episode["success"] for episode in episodes) / len(episodes)
        avg_episode_length = sum(
            int(episode["episode_length"]) for episode in episodes
        ) / len(episodes)
        avg_final_distance = sum(
            float(episode["final_distance"]) for episode in episodes
        ) / len(episodes)
        overall = {
            "success_rate": success_rate,
            "n_episodes": len(episodes),
            "avg_episode_length": avg_episode_length,
            "avg_final_distance": avg_final_distance,
        }
        results_path = output_dir / "results.json"
        results_path.write_text(
            json.dumps({"overall": overall, "episodes": episodes}, indent=2) + "\n",
            encoding="utf-8",
        )
        return EvalResult(
            overall=overall,
            per_task={task: overall},
            artifacts={"results_json": str(results_path)},
            metadata={
                "env": self.contract.env_type,
                "target": target.astype(float).tolist(),
            },
        )


def register_point_reach_driver(
    name: str = "point_reach",
    *,
    replace: bool = True,
) -> str:
    """Register the example driver and return the registered name."""

    register_driver(name, PointReachDriver(), replace=replace)
    return name


def _initial_position(config: EvalConfig, rng: np.random.Generator) -> np.ndarray:
    if "start" in config.env_kwargs:
        return _vector2(config.env_kwargs["start"], name="start")
    return rng.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)


def _vector2(value: object, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {tuple(vector.shape)}")
    return vector
