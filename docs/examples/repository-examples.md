# Runnable Repository Examples

The
[`examples/`](https://github.com/Chaoqi-LIU/praxis-eval/tree/main/examples)
directory contains small programs that can be run directly from the repository
checkout. They are separate from the benchmark setup docs: the examples teach
the Python API, while setup pages cover simulator assets and runtime
environments.

The examples use a toy point-reaching driver, so they do not require LIBERO,
RoboCasa, RoboMimic, MetaWorld, SimplerEnv, or MS-HAB assets.

## Install

Clone the repository and install the base package in editable mode:

```bash
git clone https://github.com/Chaoqi-LIU/praxis-eval
cd praxis-eval
python -m pip install -e .
```

For the remote-policy example, install the remote extra:

```bash
python -m pip install -e ".[remote]"
```

## Example Map

| Folder | What It Shows |
| --- | --- |
| [`examples/01_minimal_local_policy`](https://github.com/Chaoqi-LIU/praxis-eval/tree/main/examples/01_minimal_local_policy) | A local policy object implementing `reset(...)` and `act(...)`. |
| [`examples/02_remote_policy`](https://github.com/Chaoqi-LIU/praxis-eval/tree/main/examples/02_remote_policy) | A separate policy server called through `RemotePolicy`. |
| [`examples/03_custom_policy_adapter`](https://github.com/Chaoqi-LIU/praxis-eval/tree/main/examples/03_custom_policy_adapter) | An adapter around a model-shaped API. |
| [`examples/04_custom_benchmark_driver`](https://github.com/Chaoqi-LIU/praxis-eval/tree/main/examples/04_custom_benchmark_driver) | A custom `EvalDriver`, `EnvContract`, and `EvalResult`. |

## Shared Toy Driver

All examples except `04_custom_benchmark_driver` use
[`examples/_support/point_reach_driver.py`](https://github.com/Chaoqi-LIU/praxis-eval/blob/main/examples/_support/point_reach_driver.py).
The driver emits a flat observation:

```python
{
    "task": "move the point to the target",
    "observation.state": position,
    "observation.goal": target,
    "metadata.step": step,
}
```

and expects a batched `float32` action with per-row shape `(2,)` and values in
`[-1, 1]`.

The driver writes `results.json` under `EvalConfig.output_dir` and returns an
`EvalResult` with aggregate metrics and artifact paths.

The shared driver accepts these `EvalConfig.env_kwargs`:

| Key | Default | Meaning |
| --- | --- | --- |
| `start` | random point in `[-1, 1]^2` | Initial xy position. |
| `target` | `(0.0, 0.0)` | Goal exposed to the policy as `observation.goal`. |
| `action_scale` | `0.25` | Multiplier applied to each normalized action. |
| `max_steps` | `20` | Maximum steps per episode. |
| `success_threshold` | `0.05` | Distance to goal counted as success. |

## Local Policy

Run:

```bash
python examples/01_minimal_local_policy/run.py
```

The policy in
[`random_policy.py`](https://github.com/Chaoqi-LIU/praxis-eval/blob/main/examples/01_minimal_local_policy/random_policy.py)
consumes the full `praxis_eval.Policy` signature:

```python
def act(
    self,
    observations,
    *,
    action_spec=None,
    policy_kwargs=None,
    episode_ids=None,
):
    ...
```

This is the most direct way to receive the evaluator-provided `ActionSpec`.
Pass a full `praxis_eval.Policy` object directly when it needs `action_spec`.
Use `LocalPolicy` for simple callables or adapters that ignore `action_spec`
and only need output validation.

## Remote Policy

Start the server:

```bash
python examples/02_remote_policy/serve_policy.py --host 127.0.0.1 --port 50051
```

Run evaluation from another terminal:

```bash
python examples/02_remote_policy/run_eval.py --address 127.0.0.1:50051
```

The server implements `praxis_remote.PolicyHandler.predict_action(...)`; the
evaluator uses `praxis_eval.RemotePolicy`. The benchmark driver and evaluator do
not need to know where the policy process runs.

Remote transport preserves observations, `policy_kwargs`, and `episode_ids`.
`action_spec` stays evaluator-side validation metadata and is not sent to the
remote policy server.

## Custom Policy Adapter

Run:

```bash
python examples/03_custom_policy_adapter/run.py
```

[`fake_model.py`](https://github.com/Chaoqi-LIU/praxis-eval/blob/main/examples/03_custom_policy_adapter/fake_model.py)
exposes a model-shaped API:

```python
model.predict(task=..., state=..., target=...)
```

[`adapter.py`](https://github.com/Chaoqi-LIU/praxis-eval/blob/main/examples/03_custom_policy_adapter/adapter.py)
owns the translation between `praxis-eval` observations and that model API.
This is the pattern to use for checkpoint-backed policies from a training
codebase.

## Custom Benchmark Driver

Run:

```bash
python examples/04_custom_benchmark_driver/run.py
```

[`driver.py`](https://github.com/Chaoqi-LIU/praxis-eval/blob/main/examples/04_custom_benchmark_driver/driver.py)
defines the benchmark-side contract:

```python
class LineReachDriver:
    @property
    def contract(self): ...

    def evaluate(self, *, policy, config): ...
```

Register the driver before calling `evaluate(...)`:

```python
register_driver("line_reach", LineReachDriver())
result = evaluate("line_reach", policy=..., config=...)
```

`LineReachDriver` accepts these `EvalConfig.env_kwargs`:

| Key | Default | Meaning |
| --- | --- | --- |
| `start` | `-1.0` | Initial scalar position. |
| `target` | `1.0` | Goal exposed to the policy as `observation.goal`. |
| `step_scale` | `0.25` | Multiplier applied to each normalized action. |
| `max_steps` | `20` | Maximum steps per episode. |
| `success_threshold` | `0.05` | Distance to goal counted as success. |

For production benchmarks, keep policy-specific logic out of the driver. The
driver owns environment rollout and contracts; callers own policy adaptation.
