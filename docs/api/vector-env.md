# Vector Env

`make_vector_env` exposes the environments behind `evaluate()` as a gymnasium
`VectorEnv`, so RL code can collect rollouts with the same simulators, task
setup, and observation processing that evaluation uses.

```python
import numpy as np

from praxis_eval import make_vector_env

env = make_vector_env(
    "libero",
    lanes=[
        ("libero_spatial", 0),
        ("libero_spatial", 0),
        ("libero_spatial", 1),
        ("libero_spatial", 1),
    ],
    env_kwargs={"init_states": False},  # random initial states for training
    seed=0,
)
obs, info = env.reset()
for _ in range(1000):
    actions = np.random.uniform(-1, 1, (env.num_envs, 7)).astype(np.float32)
    obs, rewards, terminated, truncated, info = env.step(actions)
    if "final_observation" in info:
        done = info["_final_observation"]
        last_obs = info["final_observation"]  # e.g. to bootstrap truncated lanes
env.close()
```

## `make_vector_env`

```python
def make_vector_env(
    env: str,
    *,
    lanes: Sequence[tuple[str, int]],
    env_kwargs: Mapping[str, Any] | None = None,
    episode_length: int | None = None,
    seed: int = 0,
    debug_verbose: bool = False,
) -> EvalVectorEnv:
    ...
```

`env` is a driver name whose benchmark has an eval pool: `libero`, `metaworld`,
`robocasa`, `robocasa_gr1`, or `robomimic`. `env_kwargs` is the benchmark config,
the same mapping as `EvalConfig.env_kwargs`.

Each lane runs one `(task_group, task_id)` task for its whole lifetime. Lanes may
repeat a task or mix tasks. Valid pairs are the ones evaluation iterates over:

```python
from praxis_eval.envs.factory import list_tasks

lanes = list_tasks({"type": "libero", "task": "libero_spatial"})
```

The returned `EvalVectorEnv` has `num_envs`, `single_action_space`,
`action_space`, `lanes`, `lane_jobs`, `reset(seed=None)`, `step(actions)`, and
`close()`.

## Observations

Observations are the values `evaluate()` passes to a policy, batched over lanes:

- array values are numpy arrays with a leading `num_envs` axis, with the same
  keys, dtypes, and layouts a policy sees per example;
- `task` is a list with one instruction string per lane;
- scalar metadata stays a single value shared by all lanes.

Evaluation and the vector env build these with the same functions, so they cannot
drift apart. See the benchmark pages for each family's keys.

## Actions

`step` takes a float32 array of shape `(num_envs, action_dim)`. Actions are
checked against the benchmark `ActionSpec` shape, dtype, and finiteness, and
out-of-range values are clipped to the action space, as in evaluation.

## Episodes

A lane is done when its environment terminates (success or failure) or after
`episode_length` steps, which is reported in `truncated`. The default
`episode_length=None` uses each task's own horizon, the one `evaluate()` uses.

Done lanes start their next episode inside the same `step()` call:

- `obs` holds the new episode's first observation for done lanes;
- `info["final_observation"]` holds the processed last observation of every lane,
  and `info["_final_observation"]` marks the lanes that finished;
- `info["is_success"]` is a boolean array, true where a lane finished with success.

## Initial States And Seeds

Episodes are numbered the way the evaluation runner numbers them. The k-th
episode of a task runs `EvalLaneJob(task_group, task_id, k, k * n_tasks + t)`,
where `t` is the task's position among the distinct lane tasks, and resets with
seed `seed + episode_index`. Each benchmark maps that job to initial states and
scenes exactly as it does during evaluation, for example LIBERO's fixed initial
states when `init_states=True`. `env.lane_jobs` shows the job of each lane's
current episode.

`reset(seed=s)` re-bases the seeds and restarts the episode counters, so it
replays the episodes of a fresh env built with `seed=s`.
