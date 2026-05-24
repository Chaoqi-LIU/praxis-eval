from __future__ import annotations

import sys
from pathlib import Path

from random_policy import RandomPolicy

from praxis_eval import EvalConfig, evaluate, get_driver


def main() -> None:
    env_name = _register_point_reach_driver()
    driver = get_driver(env_name)
    print(driver.contract)

    result = evaluate(
        env_name,
        policy=RandomPolicy(seed=42),
        config=EvalConfig(
            task="move the point to the target",
            num_eval_per_task=3,
            output_dir=".tmp/praxis_eval_examples/01_minimal_local_policy",
            env_kwargs={"target": (0.25, -0.25), "max_steps": 8},
        ),
    )

    print("overall:", dict(result.overall))
    print("artifacts:", dict(result.artifacts))


def _register_point_reach_driver() -> str:
    examples_root = Path(__file__).resolve().parents[1]
    if str(examples_root) not in sys.path:
        sys.path.insert(0, str(examples_root))

    from _support.point_reach_driver import register_point_reach_driver

    return register_point_reach_driver()


if __name__ == "__main__":
    main()
