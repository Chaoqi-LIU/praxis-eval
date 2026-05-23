# Contracts API

Contracts describe the evaluator-policy boundary.

## `ObservationKey`

```python
@dataclass(frozen=True)
class ObservationKey:
    key: str
    dtype: str
    shape: tuple[int | str, ...] | None = None
    description: str = ""
```

Use `ObservationKey` to document one policy-facing observation key emitted by a benchmark.

## `ActionSpec`

```python
@dataclass(frozen=True)
class ActionSpec:
    shape: tuple[int, ...] | None
    dtype: str = "float32"
    minimum: float | None = None
    maximum: float | None = None
    convention: str = ""
    description: str = ""
```

`ActionSpec.validate(action)` checks one action. `ActionSpec.validate_batch(actions, batch_size=...)` checks a batched action array.

Validation covers:

- expected shape when `shape` is not `None`;
- exact numpy dtype;
- finite values;
- lower and upper bounds when configured.

## `EnvContract`

```python
@dataclass(frozen=True)
class EnvContract:
    env_type: str
    observation_keys: tuple[ObservationKey, ...]
    action: ActionSpec
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
```

Each built-in benchmark exposes an `EnvContract` through its driver:

```python
from praxis_eval import get_driver

contract = get_driver("mshab").contract
print(contract.observation_keys)
print(contract.action)
```

Contracts should describe what policy adapters receive and return. They should not document private simulator structures unless those structures are part of the public policy-facing observation mapping.
