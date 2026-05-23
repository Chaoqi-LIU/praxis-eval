# Quickstart

Install the core package and one benchmark extra:

```bash
pip install praxis-eval
pip install "praxis-eval[libero]"
```

Inspect the benchmark contract before writing a policy adapter:

```python
from praxis_eval import get_driver

driver = get_driver("libero")
print(driver.contract.observation_keys)
print(driver.contract.action)
```

Run evaluation with a local policy:

```python
import numpy as np

from praxis_eval import EvalConfig, LocalPolicy, evaluate


class ZeroPolicy:
    def reset(self, episode_ids=None) -> None:
        pass

    def act(self, observations, *, action_spec=None, policy_kwargs=None, episode_ids=None):
        del policy_kwargs, episode_ids
        if action_spec is None or action_spec.shape is None:
            raise ValueError("Expected an ActionSpec from the benchmark driver.")
        return np.zeros(
            (len(observations), *action_spec.shape),
            dtype=action_spec.dtype,
        )


result = evaluate(
    "libero",
    policy=LocalPolicy(ZeroPolicy()),
    config=EvalConfig(
        task="libero_10",
        task_ids=(0,),
        num_eval_per_task=1,
        num_parallel_env=1,
        output_dir="eval/libero_10_task0",
    ),
)

print(result.overall)
print(result.artifacts["results_path"])
```

For remote serving, install the remote extra and pass a `RemotePolicy`:

```bash
pip install "praxis-eval[remote]"
```

```python
from praxis_eval import EvalConfig, RemotePolicy, evaluate

result = evaluate(
    "robocasa",
    policy=RemotePolicy("127.0.0.1:50051", timeout=30.0),
    config=EvalConfig(
        task="CloseToasterOvenDoor",
        num_eval_per_task=5,
        output_dir="eval/robocasa_close_toaster",
    ),
)
```

Use `praxis-eval-verify` only on a machine with the required simulator runtime:

```bash
praxis-eval-verify --help
praxis-eval-verify libero --task libero_10 --task-id 0
```
