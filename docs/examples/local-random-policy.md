# Local Random Policy

This example is useful for smoke testing an installed benchmark on a simulator-capable machine. It samples within the benchmark `ActionSpec`.

```python
from pathlib import Path

import numpy as np

from praxis_eval import EvalConfig, LocalPolicy, get_driver, evaluate


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, episode_ids=None) -> None:
        pass

    def act(self, observations, *, action_spec=None, policy_kwargs=None, episode_ids=None):
        del policy_kwargs, episode_ids
        if action_spec is None or action_spec.shape is None:
            raise ValueError("RandomPolicy requires a fixed-shape ActionSpec.")

        low = -1.0 if action_spec.minimum is None else float(action_spec.minimum)
        high = 1.0 if action_spec.maximum is None else float(action_spec.maximum)
        action = self.rng.uniform(
            low=low,
            high=high,
            size=(len(observations), *action_spec.shape),
        )
        return action.astype(action_spec.dtype)


driver = get_driver("metaworld")
print(driver.contract.action)

result = evaluate(
    "metaworld",
    policy=LocalPolicy(RandomPolicy(seed=42)),
    config=EvalConfig(
        task="reach-v3",
        num_eval_per_task=1,
        num_parallel_env=1,
        output_dir=Path("eval/metaworld_random"),
    ),
)

print(result.overall)
```

Random policy rollouts do not validate benchmark quality. They validate package importability, environment construction, action shape, stepping, metrics, and artifact writing.
