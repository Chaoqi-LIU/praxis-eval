# Policies API

Policies receive a sequence of observation mappings and return a batched numpy action array.

## Policy Protocol

```python
class Policy(Protocol):
    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        ...

    def act(
        self,
        observations: Sequence[Observation],
        *,
        action_spec: ActionSpec | None = None,
        policy_kwargs: Mapping[str, Any] | None = None,
        episode_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        ...
```

`action_spec` is evaluator-side validation metadata. Policy adapters should not require underlying model code to consume it directly.

## `LocalPolicy`

```python
from praxis_eval import LocalPolicy

policy = LocalPolicy(my_policy)
```

`my_policy` may be:

- an object with `act(observations, *, policy_kwargs=None, episode_ids=None)`;
- a callable with the same signature.

If `my_policy` defines `reset(...)`, `LocalPolicy` forwards resets.

## `RemotePolicy`

```python
from praxis_eval import RemotePolicy

policy = RemotePolicy("127.0.0.1:50051", timeout=30.0)
```

`RemotePolicy` calls a `praxis-remote` policy server. Install it with:

```bash
pip install "praxis-eval[remote]"
```

Remote mode is optional. It is useful when the simulator runtime and policy runtime need different dependency stacks.

## Action Normalization

`normalize_batched_action(...)` enforces batch size, finite values, and `ActionSpec` validation when an action spec is supplied. For a single observation, unbatched action arrays are accepted if the array shape matches the single-action spec.
