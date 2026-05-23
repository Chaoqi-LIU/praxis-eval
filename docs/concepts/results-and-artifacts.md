# Results And Artifacts

`evaluate(...)` returns an `EvalResult`:

```python
print(result.overall)
print(result.per_group)
print(result.per_task)
print(result.artifacts)
print(result.metadata)
```

## Result Shape

| Field | Contents |
| --- | --- |
| `overall` | Aggregate metrics across all evaluated tasks. |
| `per_task` | Metrics keyed by `group/task_id`. |
| `per_group` | Grouped metrics when the benchmark exposes task groups. |
| `artifacts` | Paths to `results.json`, output directory, media directory, and benchmark outputs. |
| `metadata` | Evaluation mode, environment type, and caller-provided metadata. |

Common metrics include:

- `success_rate`;
- `avg_episode_length`;
- `avg_reward`;
- `avg_sum_reward`;
- `avg_max_reward`;
- `n_episodes`;
- `eval_s`;
- `eval_ep_s`;
- `video_paths`.

MS-HAB also reports `success_once_rate`, `success_at_end_rate`, and `avg_return_per_step`.

## Artifact Layout

The default artifact layout is:

```text
output_dir/
  results.json
  media/
    <task-or-benchmark-media>
```

`results.json` includes the aggregated result payload and an `_meta` block with timestamp, seed, mode, rollout settings, and caller metadata.

When `record_episodes_per_task > 0`, videos are written below the media directory. SimplerEnv and MS-HAB may run a dedicated single-env recording pass to avoid unstable or oversized multi-env video capture.
