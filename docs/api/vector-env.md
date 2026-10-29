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
    if "final_obs" in info:
        for lane in np.flatnonzero(info["_final_obs"]):
            last_obs = info["final_obs"][lane]  # e.g. to bootstrap a truncated lane
            success = info["final_info"]["is_success"][lane]
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

The returned `EvalVectorEnv` has `num_envs`, `single_observation_space`,
`observation_space`, `single_action_space`, `action_space`, `lanes`,
`lane_jobs`, `reset(seed=None)`, `step(actions)`, and `close()`.

## Observations

Observations are the values `evaluate()` passes to a policy, batched over lanes.
Evaluation and the vector env build them with the same functions, so they cannot
drift apart. An observation holds three kinds of values:

| Kind | LIBERO defaults | In `observation_space` |
| --- | --- | --- |
| numpy arrays with a leading `num_envs` axis | `observation.images.image` and `observation.images.image2`, `(num_envs, 3, 360, 360)` float32 in `[0, 1]`; `observation.state`, `(num_envs, 8)` float32 | yes |
| `task`, a list with one instruction string per lane | `["pick up the black bowl ...", ...]` | no |
| scalar metadata, one value shared by all lanes | `next.reward`, `next.done`, `next.truncated` | no |

`single_observation_space` and `observation_space` are `gymnasium.spaces.Dict`
spaces of `Box` spaces over the array keys, with the processed per-lane and
batched shapes and dtypes. They are built when the env is constructed, by
processing a zero observation of the benchmark's raw observation space with the
same functions, so they need no per-benchmark code. Their bounds are not
tightened: float arrays are unbounded and integer arrays span their dtype.

The other keys have no space. `task` is not a gymnasium `Text` space because
`Text` admits only a fixed character set up to a fixed length, so
`contains` would reject valid instructions. Check an observation against the space
on its array keys:

```python
arrays = {key: obs[key] for key in env.observation_space.spaces}
assert env.observation_space.contains(arrays)
```

`next.reward`, `next.done`, and `next.truncated` are added to every observation
by LeRobot's processor pipeline (`transition_to_batch`), and `evaluate()` passes
them to policies as well, so both keep them. They are placeholders with the
constant values `0.0`, `False`, and `False`, not the step's reward or episode end:
use the `rewards`, `terminated`, and `truncated` that `step()` returns.

## Actions

`step` takes a float32 array of shape `(num_envs, action_dim)`. Actions are
checked against the benchmark `ActionSpec` shape, dtype, and finiteness, and
out-of-range values are clipped to the action space, as in evaluation.

## Episodes

A lane is done when its environment terminates (success or environment failure),
reported in `terminated`, or when it reaches its horizon, reported in
`truncated`. The horizon is `episode_length` steps. The default
`episode_length=None` uses each task's own horizon, the one `evaluate()` uses.

Done lanes start their next episode inside the same `step()` call, and `info` has
the layout of gymnasium's `SyncVectorEnv` with `autoreset_mode=SAME_STEP`. The env
builds it lane by lane with gymnasium's own `VectorEnv._add_info`, so the
batching and masks are the installed gymnasium's.

- `obs` holds the new episode's first observation for done lanes.
- `info["final_obs"]` is an object array of length `num_envs`. A done lane's entry
  is its last observation, a per-lane dict as `Policy.act` receives it; the other
  entries are `None`. `info["_final_obs"]` marks the done lanes.
- `info["final_info"]` holds the done lanes' last step info, batched with a
  `_<key>` mask per key, and `info["_final_info"]` marks the done lanes. It is
  filled for every done lane, including lanes truncated by `episode_length`.
  `info["final_info"]["is_success"]` is the success `evaluate()` records for the
  episode.
- The other `info` keys hold the step info of lanes that continue and the reset
  info of lanes that restarted.

`final_obs` and `final_info` appear only on steps where some lane is done.

## Simulator Coverage

`make_vector_env` has run on real simulators for LIBERO, MetaWorld, RoboMimic,
and RoboCasa, with random actions through at least one truncation at each
benchmark's own horizon. RoboCasa GR-1 has not run on a real simulator. Success
and environment termination are covered only by unit tests, because random
actions do not succeed.

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
