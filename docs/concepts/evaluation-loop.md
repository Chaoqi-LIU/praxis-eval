# Evaluation Loop

Evaluation starts with:

```python
from praxis_eval import EvalConfig, LocalPolicy, evaluate

policy = LocalPolicy(my_policy)
config = EvalConfig(task="libero_10", num_eval_per_task=5, output_dir="eval/libero")

result = evaluate("libero", policy=policy, config=config)
```

`evaluate(...)` resolves the benchmark driver, builds benchmark-specific environment config from `EvalConfig`, prepares artifact directories, runs rollouts, writes `results.json`, and returns an `EvalResult`.

## Driver Dispatch

Built-in drivers are registered lazily. A benchmark driver exposes:

- `contract`: an `EnvContract` containing observation keys and an `ActionSpec`.
- `evaluate(policy, config)`: the method that runs rollout orchestration.

Most current-environment benchmarks use the async pool path:

1. Resolve tasks from the benchmark-specific config.
2. Build a persistent environment pool.
3. Reset policy state.
4. Send batched observations to the policy adapter.
5. Validate batched actions with `ActionSpec`.
6. Step environments, collect terminal metrics, and optionally record videos.
7. Aggregate overall, per-group, and per-task metrics.

SimplerEnv and MS-HAB use a subprocess path. The evaluator starts or uses a `praxis-remote` policy endpoint, runs the external simulator command in the dedicated runtime, then reads task metrics and artifacts from disk.

## EvalConfig

`EvalConfig` contains generic evaluation settings:

| Field | Purpose |
| --- | --- |
| `num_eval_per_task` | Number of episodes per resolved task. |
| `output_dir` | Directory for results and media. |
| `task` | Benchmark task selector. |
| `task_ids` | Optional task subset after task resolution. |
| `num_parallel_env` | Parallel environment lanes for each wave. |
| `seed` | Start seed for deterministic episode indexing. |
| `record_episodes_per_task` | Number of videos to save per task. |
| `step_timeout_sec` | Optional rollout step timeout. |
| `rollout_failure_retries` | Retry count for rollout failures. |
| `policy_kwargs` | Extra kwargs forwarded to the policy adapter. |
| `env_kwargs` | Benchmark-specific environment settings. |

Benchmark-specific options belong in `env_kwargs`, not in the policy adapter.
