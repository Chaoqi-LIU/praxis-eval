from __future__ import annotations

import sys
from pathlib import Path

from adapter import PointModelPolicyAdapter
from fake_model import FakePointModel

from praxis_eval import EvalConfig, evaluate


def main() -> None:
    env_name = _register_point_reach_driver()
    result = evaluate(
        env_name,
        policy=PointModelPolicyAdapter(FakePointModel()),
        config=EvalConfig(
            task="move the point to the target",
            num_eval_per_task=2,
            output_dir=".tmp/praxis_eval_examples/03_custom_policy_adapter",
            policy_kwargs={"gain": 1.0},
            env_kwargs={
                "start": (0.8, -0.8),
                "target": (0.0, 0.0),
                "max_steps": 16,
            },
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
