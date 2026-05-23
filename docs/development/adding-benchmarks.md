# Adding Benchmarks

This guide describes how to add a benchmark family to `praxis-eval`.

Keep the boundary clear: `praxis-eval` owns benchmark setup, task resolution, environment construction, rollout, metrics, artifacts, and observation/action contracts. Policy-specific preprocessing, checkpoint loading, model code, and training infrastructure stay outside the evaluator.

## Choose A Runtime Pattern

Use the current-environment async pool pattern when the simulator can run in the same Python environment as `praxis-eval`.

Use the subprocess pattern when the simulator needs a dedicated runtime or incompatible dependencies. SimplerEnv and MS-HAB are examples. In that pattern, the evaluator communicates with the policy through `praxis-remote` and launches task-specific external commands.

## Files To Add

A typical current-environment benchmark lives under `src/praxis_eval/envs/<name>/`:

```text
src/praxis_eval/envs/<name>/
  __init__.py        # EnvContract and lazy public exports
  config.py          # dataclass env config and feature metadata
  tasks.py           # task selectors, aliases, dataset inference
  env.py             # gymnasium wrapper, if evaluator-owned
  runtime.py         # worker-local wrapper and dummy env for spaces
  eval.py            # persistent eval pool builder
  processor.py       # optional env pre/post processors
  registration.py    # factory registration hook
```

A subprocess-backed benchmark usually replaces `env.py`, `runtime.py`, and the eval pool builder with command construction and metrics aggregation in `eval.py`.

## Define The Contract

Start with `EnvContract` in `__init__.py`:

```python
from praxis_eval.contracts import ActionSpec, EnvContract, ObservationKey

CONTRACT = EnvContract(
    env_type="mybench",
    observation_keys=(
        ObservationKey("task", "str", description="Task instruction."),
        ObservationKey(
            "observation.images.front",
            "float32",
            shape=(3, 128, 128),
            description="Front RGB image.",
        ),
        ObservationKey("observation.state", "float32", shape=(10,)),
    ),
    action=ActionSpec(
        shape=(7,),
        dtype="float32",
        minimum=-1.0,
        maximum=1.0,
        convention="mybench_normalized_action",
    ),
)
```

The contract should document policy-facing observations, not raw simulator internals. If the wrapper emits raw HWC images but the processor sends CHW images to policies, document the CHW policy-facing shape.

## Add Config

Create a dataclass config with `type` and `task` fields:

```python
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class MybenchEnvConfig:
    processor_factory: ClassVar[str] = "identity"

    type: str = "mybench"
    task: str = "default_task"
    image_size: int = 128
    max_episode_steps: int = 500
```

If the benchmark needs env preprocessing, set `processor_factory` to `"module:function"` and return the env pre/post processor pair from that function.

## Resolve Tasks

Task listers return `(task_group, task_id)` pairs:

```python
def list_mybench_tasks(raw_cfg, cfg_obj, debug_verbose=False):
    del raw_cfg, debug_verbose
    if cfg_obj.task == "default_task":
        return [("default_task", 0)]
    return [(str(cfg_obj.task), 0)]
```

Also add dataset-name inference when dataset names imply an evaluation target:

```python
def infer_mybench_eval_target_from_dataset(dataset_name: str):
    if dataset_name.startswith("mybench_"):
        return "mybench", dataset_name.removeprefix("mybench_")
    return None
```

## Current-Environment Runtime

For async pool benchmarks, provide:

- a dummy env that exposes observation and action spaces without initializing the simulator;
- a real env factory for workers;
- a worker wrapper with `prepare_eval_job(...)`;
- an eval pool builder registered with the factory.

The existing LIBERO, RoboCasa, RoboMimic, and MetaWorld drivers are the best references.

Register the family:

```python
def register_mybench_env_family() -> None:
    from praxis_eval.envs.factory import (
        register_env_config,
        register_eval_pool_builder,
        register_eval_target_inferer,
        register_task_lister,
    )

    register_env_config("mybench", "praxis_eval.envs.mybench.config:MybenchEnvConfig")
    register_task_lister("mybench", "praxis_eval.envs.mybench.tasks:list_mybench_tasks")
    register_eval_target_inferer(
        "mybench",
        "praxis_eval.envs.mybench.tasks:infer_mybench_eval_target_from_dataset",
    )
    register_eval_pool_builder(
        "mybench",
        "praxis_eval.envs.mybench.eval:build_mybench_eval_pool",
    )
```

Then add the registrar to `_DEFAULT_ENV_FAMILY_REGISTRARS` in `praxis_eval.envs.factory` and add the contract to `register_builtin_contract_drivers()` in `praxis_eval.envs.builtins`.

## Subprocess Runtime

For dedicated runtimes:

1. Define task specs and task selector expansion.
2. Resolve `python_bin` from env config.
3. Build an external command for one task.
4. Write metrics to a JSON path supplied by the evaluator.
5. Summarize subprocess task results into the standard `overall`, `per_group`, and `per_task` shape.
6. Register a runtime driver with `register_env_runtime_driver("mybench", "module:function")`.

The external runtime should receive observations from its simulator, build policy-facing mappings, call the policy through `praxis-remote`, and apply returned actions to the simulator.

## Setup And Verify Commands

Add setup only when the benchmark needs assets or a dedicated runtime:

```text
src/praxis_eval/scripts/setup_mybench.py
```

Register it in `SETUP_MODULES` in `praxis_eval.scripts.setup`.

Add a verifier that runs a short random-action rollout:

```text
src/praxis_eval/scripts/verify_mybench.py
```

Register it in `VERIFY_MODULES` in `praxis_eval.scripts.verify`.

Verifiers are for simulator-capable machines. They should expose `--help`, write `results.json`, and use shared helpers from `praxis_eval.scripts._verify_common`.

## Tests

Add focused tests that do not require the simulator:

- contract registration and `available_drivers()`;
- config default values and feature maps;
- task selector expansion and aliases;
- action shape validation;
- observation conversion for policy-facing keys;
- setup/verify `--help` importability without simulator extras where possible;
- subprocess command construction if using a dedicated runtime.

Simulator rollouts should be marked or isolated so normal CI can run without heavyweight benchmark stacks.

## Documentation Checklist

Every new benchmark page should include:

- install extra;
- setup command, if any;
- verify command;
- default task examples;
- observation keys with shapes and dtypes;
- action shape, dtype, range, and convention;
- runtime model: current environment or dedicated runtime;
- assets, OpenGL, runtime, and dependency caveats.
