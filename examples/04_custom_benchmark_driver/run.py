from __future__ import annotations

import numpy as np
from driver import LineReachDriver

from praxis_eval import EvalConfig, LocalPolicy, evaluate, register_driver


def act(observations, *, policy_kwargs=None, episode_ids=None):
    del policy_kwargs, episode_ids
    actions = []
    for observation in observations:
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        target = np.asarray(observation["observation.goal"], dtype=np.float32)
        actions.append(np.clip(target - state, -1.0, 1.0))
    return np.stack(actions).astype(np.float32)


def main() -> None:
    register_driver("line_reach", LineReachDriver())
    result = evaluate(
        "line_reach",
        policy=LocalPolicy(act),
        config=EvalConfig(
            task="move right on the line",
            num_eval_per_task=1,
            output_dir=".tmp/praxis_eval_examples/04_custom_benchmark_driver",
            env_kwargs={"start": -1.0, "target": 1.0},
        ),
    )

    print("overall:", dict(result.overall))
    print("artifacts:", dict(result.artifacts))


if __name__ == "__main__":
    main()
