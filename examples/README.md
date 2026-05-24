# Examples

These examples are small, runnable programs that show the main
`praxis-eval` extension points:

- a policy receives a batch of observation dictionaries
- a policy returns a batched `numpy.ndarray` action
- an evaluation driver owns the observation/action contract
- local and remote policy execution use the same evaluator API

The examples use a toy point-reaching driver in
[`_support/point_reach_driver.py`](_support/point_reach_driver.py), so they can
run without simulator assets. Real benchmark setup stays in the documentation;
this directory is for API usage patterns you can execute quickly.

Clone the repository and install the package in editable mode before running
the local examples:

```bash
git clone https://github.com/Chaoqi-LIU/praxis-eval
cd praxis-eval
python -m pip install -e .
```

For the remote-policy example, also install the remote extra:

```bash
python -m pip install -e ".[remote]"
```

Run examples from the repository root:

```bash
python examples/01_minimal_local_policy/run.py
python examples/03_custom_policy_adapter/run.py
python examples/04_custom_benchmark_driver/run.py
```

The remote example uses two processes:

```bash
python examples/02_remote_policy/serve_policy.py
python examples/02_remote_policy/run_eval.py
```

## Contents

- [`01_minimal_local_policy`](01_minimal_local_policy): evaluate an in-process
  Python policy object that implements the full `praxis_eval.Policy` protocol.
- [`02_remote_policy`](02_remote_policy): serve a policy with `praxis-remote`
  and call it through `praxis_eval.RemotePolicy`.
- [`03_custom_policy_adapter`](03_custom_policy_adapter): adapt an existing
  model API to the `praxis_eval.Policy` contract.
- [`04_custom_benchmark_driver`](04_custom_benchmark_driver): implement and
  register a small custom benchmark driver.

## Shared Toy Driver

[`_support/point_reach_driver.py`](_support/point_reach_driver.py) defines a
minimal `EvalDriver`. It documents the observation keys:

- `task`
- `observation.state`
- `observation.goal`
- `metadata.step`

and the action contract:

- shape `(2,)`
- dtype `float32`
- values in `[-1, 1]`

The driver writes a `results.json` artifact under the configured `output_dir`.
Each example prints the aggregate metrics and artifact path returned in
`EvalResult`.

The shared driver accepts these `EvalConfig.env_kwargs`:

| Key | Default | Meaning |
| --- | --- | --- |
| `start` | random point in `[-1, 1]^2` | Initial xy position. |
| `target` | `(0.0, 0.0)` | Goal exposed to the policy as `observation.goal`. |
| `action_scale` | `0.25` | Multiplier applied to each normalized action. |
| `max_steps` | `20` | Maximum steps per episode. |
| `success_threshold` | `0.05` | Distance to goal counted as success. |

Pass a full `praxis_eval.Policy` object directly when it needs `action_spec`.
Use `LocalPolicy` for simple callables or adapters that ignore `action_spec`
and only need output validation.
