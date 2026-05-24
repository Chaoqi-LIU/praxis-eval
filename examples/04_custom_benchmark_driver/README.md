# 04 Custom Benchmark Driver

This example implements and registers a complete custom `EvalDriver`. A driver
is the benchmark-side boundary: it documents observations and actions, runs the
environment, calls the policy, writes artifacts, and returns `EvalResult`.

## Files

- [`driver.py`](driver.py) defines `LineReachDriver`, its `EnvContract`, and
  its evaluation loop.
- [`run.py`](run.py) registers the driver with `register_driver(...)` and calls
  `evaluate(...)`.

## Run

From the repository root:

```bash
python examples/04_custom_benchmark_driver/run.py
```

The script writes artifacts under:

```text
.tmp/praxis_eval_examples/04_custom_benchmark_driver/
```

## What To Notice

[`driver.py`](driver.py) exposes the same interface as built-in drivers:

```python
class LineReachDriver:
    @property
    def contract(self) -> EnvContract: ...

    def evaluate(self, *, policy: Policy, config: EvalConfig) -> EvalResult: ...
```

The `contract` documents the observation keys and `ActionSpec` before any
rollout runs. The `evaluate(...)` method owns benchmark-specific rollout
details and reads benchmark settings from `config.env_kwargs`.

`LineReachDriver` accepts these `EvalConfig.env_kwargs`:

| Key | Default | Meaning |
| --- | --- | --- |
| `start` | `-1.0` | Initial scalar position. |
| `target` | `1.0` | Goal exposed to the policy as `observation.goal`. |
| `step_scale` | `0.25` | Multiplier applied to each normalized action. |
| `max_steps` | `20` | Maximum steps per episode. |
| `success_threshold` | `0.05` | Distance to goal counted as success. |

For a production benchmark, keep the same boundary:

- document observations with `EnvContract.observation_keys`
- document expected actions with `EnvContract.action`
- put benchmark-specific settings in `EvalConfig.env_kwargs`
- pass policy-side options through `EvalConfig.policy_kwargs`
- return aggregate metrics and artifact paths in `EvalResult`

Keep policy adaptation outside the driver. The driver should know the action
contract it expects, but it should not know how a caller's model is implemented.
