# 03 Custom Policy Adapter

Most real policies do not expose the exact `praxis_eval.Policy` method
signature. This example wraps a small model-shaped API with an adapter that
consumes `praxis-eval` observations and returns `praxis-eval` actions.

## Files

- [`fake_model.py`](fake_model.py) represents a checkpoint-backed model API. It
  knows nothing about `praxis-eval`.
- [`adapter.py`](adapter.py) maps `praxis-eval` observations into model inputs
  and maps model outputs back to eval actions.
- [`run.py`](run.py) passes the adapter into `praxis_eval.evaluate(...)`.

## Run

From the repository root:

```bash
python examples/03_custom_policy_adapter/run.py
```

## What To Notice

[`adapter.py`](adapter.py) is the boundary a training or inference codebase
usually owns. It handles:

- observation keys such as `observation.state` and `observation.goal`
- batching across `Sequence[Observation]`
- optional `policy_kwargs`
- action dtype and shape validation through `ActionSpec`

The model in [`fake_model.py`](fake_model.py) stays model-shaped:

```python
model.predict(task=..., state=..., target=...)
```

That separation keeps benchmark logic out of the model and model logic out of
the evaluator.
