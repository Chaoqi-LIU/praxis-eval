# 01 Minimal Local Policy

This example evaluates a local Python policy in the same process as the
evaluation driver. It is the smallest complete example of the core
`praxis-eval` contract: the evaluator sends observations, and the policy returns
actions.

It uses the toy `point_reach` driver from
[`../_support/point_reach_driver.py`](../_support/point_reach_driver.py), so it
does not need simulator assets.

## Files

- [`run.py`](run.py) registers the toy driver, builds `EvalConfig`, calls
  `praxis_eval.evaluate(...)`, and prints `EvalResult`.
- [`random_policy.py`](random_policy.py) implements a policy object with
  `reset(...)` and `act(...)`.

## Run

From the repository root:

```bash
python examples/01_minimal_local_policy/run.py
```

The script writes artifacts under:

```text
.tmp/praxis_eval_examples/01_minimal_local_policy/
```

## What To Notice

[`random_policy.py`](random_policy.py) receives:

- `observations`: a sequence of dictionaries
- `action_spec`: the action shape, dtype, and bounds from the driver
- `policy_kwargs`: caller-provided inference options
- `episode_ids`: rollout identifiers for policy-side state

This example passes `RandomPolicy` directly to `evaluate(...)` so the policy can
see `ActionSpec`. `LocalPolicy` is useful when wrapping a simple callable or
object that does not need to consume `ActionSpec` itself.

[`run.py`](run.py) uses only the public API:

```python
result = evaluate(
    env_name,
    policy=RandomPolicy(seed=42),
    config=EvalConfig(...),
)
```

The same pattern works for built-in benchmarks such as `metaworld`, `libero`,
or `robocasa` once their simulator dependencies and assets are installed.
