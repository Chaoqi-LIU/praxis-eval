# Observations And Actions

Policies receive observations as Python mappings:

```python
{
    "task": "put the carrot on the plate",
    "observation.images.image": image,
    "observation.state": proprio,
    "metadata.episode_index": 7,
}
```

Values may be numpy arrays, strings, booleans, integers, floats, or numpy scalar values. The evaluator does not know how to tokenize text, normalize model inputs, load checkpoints, or run model-specific preprocessing. Your policy adapter owns that translation.

## Common Key Conventions

| Key pattern | Meaning |
| --- | --- |
| `task` | Natural-language instruction or task label. |
| `observation.images.<name>` | RGB camera image, usually channel-first `(C, H, W)`. |
| `observation.state` | Flattened proprioceptive state. |
| `observation.state.<name>` | Named state component kept for policy IO compatibility. |
| `metadata.<name>` | Non-action rollout metadata. |

Rollout bookkeeping keys such as `action` and `info` are not forwarded to local policy observations.

## Image Dtypes And Layout

Benchmark wrappers generally convert images into policy-facing channel-first arrays. Some paths preserve `uint8`; others emit `float32`, often normalized to `[0, 1]`. Benchmark pages document the expected keys and dtypes as implemented in this repository.

Policy adapters should inspect the benchmark contract and normalize inputs for their own model. Do not assume all benchmarks use the same camera names or image dtype.

## Action Validation

A policy returns a batched `numpy.ndarray` action:

```python
def act(observations, *, action_spec=None, policy_kwargs=None, episode_ids=None):
    return np.zeros((len(observations), 7), dtype=np.float32)
```

When the benchmark supplies an `ActionSpec`, `praxis-eval` validates:

- batch size;
- per-action shape;
- dtype;
- finite values;
- minimum and maximum bounds when configured.

For a single-observation batch, a policy may return either `(action_dim,)` or `(1, action_dim)` when an `ActionSpec` is available. Multi-observation calls must return a batched array.

The evaluator validates only the documented action contract. It does not infer what a particular model or checkpoint expects.
