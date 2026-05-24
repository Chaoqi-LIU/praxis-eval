from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from praxis_eval import (
    ActionSpec,
    EnvContract,
    EvalConfig,
    EvalResult,
    ObservationKey,
)
from praxis_eval.types import Policy


class LineReachDriver:
    """One-dimensional custom benchmark driver."""

    @property
    def contract(self) -> EnvContract:
        return EnvContract(
            env_type="line_reach",
            observation_keys=(
                ObservationKey("task", "str"),
                ObservationKey("observation.state", "float32", shape=(1,)),
                ObservationKey("observation.goal", "float32", shape=(1,)),
            ),
            action=ActionSpec(
                shape=(1,),
                dtype="float32",
                minimum=-1.0,
                maximum=1.0,
                convention="Normalized one-dimensional delta.",
            ),
            notes="Example-only driver for adding a new benchmark family.",
        )

    def evaluate(self, *, policy: Policy, config: EvalConfig) -> EvalResult:
        if config.num_eval_per_task < 1:
            raise ValueError("num_eval_per_task must be >= 1")

        task = config.task or "move right on the line"
        target = _scalar_array(config.env_kwargs.get("target", 1.0), name="target")
        start = _scalar_array(config.env_kwargs.get("start", -1.0), name="start")
        step_scale = float(config.env_kwargs.get("step_scale", 0.25))
        max_steps = int(config.env_kwargs.get("max_steps", 20))
        threshold = float(config.env_kwargs.get("success_threshold", 0.05))

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        policy.reset()
        episodes = []
        for episode_index in range(config.num_eval_per_task):
            episode_id = f"line-reach/{episode_index}"
            policy.reset(episode_ids=[episode_id])
            position = start.copy()
            trace = []
            success = False
            for step in range(max_steps):
                observation = {
                    "task": task,
                    "observation.state": position.astype(np.float32),
                    "observation.goal": target.astype(np.float32),
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
                position = position + action.astype(np.float32) * step_scale
                distance = float(abs(target[0] - position[0]))
                trace.append(
                    {
                        "step": step,
                        "position": float(position[0]),
                        "action": float(action[0]),
                        "distance": distance,
                    }
                )
                if distance <= threshold:
                    success = True
                    break

            episodes.append(
                {
                    "episode_id": episode_id,
                    "success": success,
                    "episode_length": len(trace),
                    "final_distance": float(abs(target[0] - position[0])),
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
            metadata={"env": self.contract.env_type},
        )


def _scalar_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray([value], dtype=np.float32).reshape(-1)
    if array.shape != (1,):
        raise ValueError(f"{name} must be a scalar")
    return array
