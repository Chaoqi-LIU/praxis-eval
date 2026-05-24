---
layout: home

hero:
  name: "praxis-eval"
  text: "Standalone robot-policy evaluation"
  tagline: Keep benchmark setup, rollout execution, metrics, artifacts, and contracts inside the evaluator. Keep policy code in your own adapter.
  actions:
    - theme: brand
      text: Quickstart
      link: /quickstart
    - theme: alt
      text: Add a Benchmark
      link: /development/adding-benchmarks
    - theme: alt
      text: View on GitHub
      link: https://github.com/Chaoqi-LIU/praxis-eval

features:
  - title: Benchmark-Owned Rollouts
    details: Drivers own environment setup, task resolution, rollout waves, success metrics, media, and results.
  - title: Policy-Facing Contracts
    details: Observation keys and action specs are documented at the evaluator boundary instead of buried in simulator internals.
  - title: Local Or Remote Policies
    details: Run an in-process policy adapter or call a separate policy server through optional praxis-remote transport.
  - title: Dedicated Simulator Runtimes
    details: SimplerEnv and MS-HAB can run in isolated runtimes while policy dependencies stay in the caller environment.
  - title: Benchmark Families
    details: Built-in coverage includes LIBERO, RoboCasa, RoboMimic, MetaWorld, SimplerEnv, and MS-HAB.
  - title: Developer Extension Path
    details: Add new benchmarks through contracts, configs, task selectors, runtime drivers, setup tools, and focused tests.
---

## Quick Example

```python
import numpy as np

from praxis_eval import EvalConfig, evaluate


class ZeroPolicy:
    def reset(self, episode_ids=None) -> None:
        pass

    def act(self, observations, *, action_spec=None, policy_kwargs=None, episode_ids=None):
        if action_spec is None or action_spec.shape is None:
            raise ValueError("Expected a fixed-shape ActionSpec.")
        return np.zeros((len(observations), *action_spec.shape), dtype=action_spec.dtype)


result = evaluate(
    "libero",
    policy=ZeroPolicy(),
    config=EvalConfig(
        task="libero_10",
        task_ids=(0,),
        num_eval_per_task=5,
        output_dir="eval/libero",
    ),
)

print(result.overall)
print(result.artifacts)
```

## Benchmark Coverage

- **LIBERO**: current-environment evaluation with LIBERO suites and normalized 7-D actions.
- **RoboCasa**: RoboCasa365 tasks, asset setup, 16-D state, and 12-D mobile manipulation actions.
- **RoboMimic**: robosuite-backed RoboMimic tasks with task aliases, known horizons, and 7-D actions.
- **MetaWorld**: MT50 selectors, difficulty groups, pixel/state observations, and 4-D actions.
- **SimplerEnv**: Bridge tasks executed through a dedicated SimplerEnv runtime and remote policy transport.
- **MS-HAB**: set-table subtasks, RGB policy observations, and dedicated MS-HAB runtime execution.

## What To Read Next

- [Quickstart](/quickstart) for the shortest working API path.
- [Installation](/installation) for extras, setup commands, and verifier commands.
- [Observations And Actions](/concepts/observations-and-actions) for the policy contract.
- [Adding Benchmarks](/development/adding-benchmarks) for the developer extension workflow.
