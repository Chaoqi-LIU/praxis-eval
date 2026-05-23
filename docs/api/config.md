# Config API

`EvalConfig` contains benchmark-agnostic evaluation settings. Benchmark-specific settings go in `env_kwargs`; model-specific inference settings go in `policy_kwargs`.

## `EvalConfig`

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EvalConfig:
    num_eval_per_task: int
    output_dir: str | Path
    task: str | None = None
    task_ids: tuple[int, ...] | None = None
    num_parallel_env: int = 1
    seed: int = 42
    record_episodes_per_task: int = 0
    step_timeout_sec: float | None = None
    rollout_failure_retries: int = 1
    debug_verbose: bool = False
    policy_kwargs: Mapping[str, Any] = field(default_factory=dict)
    env_kwargs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_hooks: EvalRuntimeHooks = field(default_factory=EvalRuntimeHooks)
```

| Field | Meaning |
| --- | --- |
| `num_eval_per_task` | Number of episodes per resolved task. |
| `output_dir` | Directory for `results.json` and media artifacts. |
| `task` | Benchmark task selector. |
| `task_ids` | Optional subset after benchmark task expansion. |
| `num_parallel_env` | Parallel environment lanes. |
| `seed` | Starting seed for deterministic episode indexing. |
| `record_episodes_per_task` | Number of videos to save per task. |
| `step_timeout_sec` | Optional timeout for rollout steps. |
| `rollout_failure_retries` | Retry count for failed rollout waves. |
| `debug_verbose` | Let benchmark runtimes print more detail. |
| `policy_kwargs` | Extra kwargs forwarded to policy inference. |
| `env_kwargs` | Benchmark-specific options. |
| `metadata` | Caller-provided metadata written into result artifacts. |
| `runtime_hooks` | Optional heartbeat callbacks. |

## `EvalResult`

```python
@dataclass(frozen=True)
class EvalResult:
    overall: Mapping[str, Any]
    per_task: Mapping[str, Mapping[str, Any]]
    per_group: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

`overall`, `per_task`, and `per_group` contain metrics. `artifacts` contains paths such as `results_path`, `output_dir`, and `media_dir`. `metadata` includes evaluator mode and environment type.

## Config Helpers

The package also exposes small helpers for config parsing:

- `env_type_from_cfg(...)`
- `optional_env_task(...)`
- `optional_env_task_ids(...)`
- `env_kwargs_without_type_task(...)`
- `resolve_policy_kwargs(...)`

They are intended for integration code that adapts external config systems into `EvalConfig`.
