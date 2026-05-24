# 02 Remote Policy

This example runs the evaluator and policy in separate processes. It shows the
same `evaluate(...)` call as a local policy, but the policy action is requested
over `praxis-remote`.

Remote mode is useful when the simulator environment and policy runtime cannot
share one Python environment, or when the policy should run on another machine.

## Files

- [`policy_handler.py`](policy_handler.py) implements the server-side
  `praxis_remote.PolicyHandler` shape.
- [`serve_policy.py`](serve_policy.py) starts a `PolicyServer`.
- [`run_eval.py`](run_eval.py) evaluates through `praxis_eval.RemotePolicy`.

## Install

From a repository checkout, install the remote extra:

```bash
python -m pip install -e ".[remote]"
```

## Run

Start the policy server in one terminal:

```bash
python examples/02_remote_policy/serve_policy.py --host 127.0.0.1 --port 50051
```

Run evaluation in another terminal:

```bash
python examples/02_remote_policy/run_eval.py --address 127.0.0.1:50051
```

Use `127.0.0.1` for same-machine evaluation. Use a reachable hostname or an
SSH tunnel when the policy process runs elsewhere.

## What To Notice

[`policy_handler.py`](policy_handler.py) exposes `predict_action(...)`, not an
eval-specific API. It receives the same observation dictionaries the evaluator
would pass to a local policy, and it returns a batched `numpy.ndarray`.

[`run_eval.py`](run_eval.py) keeps the evaluation code small:

```python
policy = RemotePolicy("127.0.0.1:50051", timeout=10.0)
result = evaluate(env_name, policy=policy, config=EvalConfig(...))
```

The evaluator still owns `EvalConfig`, driver contracts, rollout execution,
metrics, and artifacts. The remote policy server owns only policy inference.
The transport preserves observations, `policy_kwargs`, and `episode_ids`.
`action_spec` stays evaluator-side validation metadata and is not sent to the
remote policy server.
