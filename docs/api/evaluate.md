# Evaluate API

The top-level API dispatches a policy and config to a registered benchmark driver.

```python
import numpy as np

from praxis_eval import EvalConfig, LocalPolicy, evaluate


class ZeroPolicy:
    def reset(self, episode_ids=None) -> None:
        pass

    def act(self, observations, *, action_spec=None, policy_kwargs=None, episode_ids=None):
        if action_spec is None or action_spec.shape is None:
            raise ValueError("Expected a fixed-shape ActionSpec.")
        return np.zeros((len(observations), *action_spec.shape), dtype=action_spec.dtype)


result = evaluate(
    "libero",
    policy=LocalPolicy(ZeroPolicy()),
    config=EvalConfig(
        task="libero_10",
        num_eval_per_task=5,
        output_dir="eval/libero",
    ),
)
```

## `evaluate`

```python
def evaluate(
    env: str | EvalDriver,
    *,
    policy: Policy,
    config: EvalConfig,
) -> EvalResult:
    ...
```

`env` may be a registered driver name such as `"libero"` or an explicit object implementing the `EvalDriver` protocol. String names are resolved with `get_driver(...)`.

`policy` must implement the `Policy` protocol. In most user code this is `LocalPolicy(...)` or `RemotePolicy(...)`.

`config` is an `EvalConfig` instance. It contains generic rollout settings and benchmark-specific `env_kwargs`.

## Driver Lookup

```python
from praxis_eval import available_drivers, get_driver

print(available_drivers())
driver = get_driver("robocasa")
print(driver.contract)
```

Built-in contract drivers are registered lazily on first lookup. Current built-in names are:

- `libero`
- `metaworld`
- `mshab`
- `robocasa`
- `robomimic`
- `simpler`

## Custom Drivers

```python
from praxis_eval import register_driver

register_driver("mybench", driver)
```

A custom driver must expose a `contract` property and an `evaluate(policy, config)` method returning an `EvalResult`.
