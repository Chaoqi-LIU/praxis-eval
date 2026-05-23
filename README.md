# praxis-eval

`praxis-eval` is a standalone robot-policy evaluation package for simulation benchmarks. It owns benchmark setup, rollout execution, metrics, artifacts, and observation/action contracts; users provide a policy adapter that consumes documented observations and returns documented actions.

## Installation

`praxis-eval` supports Python 3.10 and newer.

```bash
pip install praxis-eval==0.1.1
```

Install the benchmark extras you plan to run:

```bash
pip install "praxis-eval[libero]==0.1.1"
pip install "praxis-eval[robocasa]==0.1.1"
pip install "praxis-eval[robomimic]==0.1.1"
pip install "praxis-eval[metaworld]==0.1.1"
pip install "praxis-eval[simpler]==0.1.1"
pip install "praxis-eval[mshab]==0.1.1"
pip install "praxis-eval[remote]==0.1.1"
```

## Minimal Local Evaluation

This example assumes the LIBERO extra and simulator runtime are installed.

```python
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
            raise ValueError("This example needs a benchmark ActionSpec.")
        low = -1.0 if action_spec.minimum is None else action_spec.minimum
        high = 1.0 if action_spec.maximum is None else action_spec.maximum
        return self.rng.uniform(
            low=low,
            high=high,
            size=(len(observations), *action_spec.shape),
        ).astype(action_spec.dtype)


driver = get_driver("libero")
print(driver.contract)

result = evaluate(
    "libero",
    policy=LocalPolicy(RandomPolicy(seed=42)),
    config=EvalConfig(
        task="libero_10",
        task_ids=(0,),
        num_eval_per_task=1,
        num_parallel_env=1,
        output_dir="eval/libero_smoke",
    ),
)

print(result.overall)
print(result.artifacts)
```

## Supported Benchmarks

| Driver | Install extra | Runtime model |
| --- | --- | --- |
| `libero` | `praxis-eval[libero]` | Runs in the current Python environment. |
| `robocasa` | `praxis-eval[robocasa]` | Runs in the current Python environment after asset setup. |
| `robomimic` | `praxis-eval[robomimic]` | Runs in the current Python environment. |
| `metaworld` | `praxis-eval[metaworld]` | Runs in the current Python environment. |
| `simpler` | `praxis-eval[simpler]` | Uses a dedicated SimplerEnv runtime for simulator execution. |
| `mshab` | `praxis-eval[mshab]` | Uses a dedicated MS-HAB runtime for simulator execution. |

Use the setup and verification CLIs to inspect available benchmark-specific commands:

```bash
praxis-eval-setup --help
praxis-eval-verify --help
```

## Documentation

Full documentation is in [docs/](docs/index.md), including installation details, benchmark contracts, remote policy evaluation, runtime setup, examples, and the developer guide for adding new benchmarks.

## License And Citation

`praxis-eval` is licensed under Apache-2.0. If this package supports your research or product work, cite the repository using [CITATION.cff](CITATION.cff).
