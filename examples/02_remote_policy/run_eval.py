from __future__ import annotations

import argparse
import sys
from pathlib import Path

from praxis_eval import EvalConfig, RemotePolicy, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    env_name = _register_point_reach_driver()
    policy = RemotePolicy(args.address, timeout=args.timeout)
    try:
        result = evaluate(
            env_name,
            policy=policy,
            config=EvalConfig(
                task="move the point to the target",
                num_eval_per_task=2,
                output_dir=".tmp/praxis_eval_examples/02_remote_policy",
                policy_kwargs={"gain": 1.0},
                env_kwargs={
                    "start": (0.9, -0.9),
                    "target": (0.0, 0.0),
                    "max_steps": 16,
                },
            ),
        )
    finally:
        policy.close()

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
