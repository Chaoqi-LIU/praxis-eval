# praxis-eval

`praxis-eval` is a standalone robot-policy evaluation package. It owns
benchmark setup, rollout execution, metrics, artifacts, and documented
environment contracts. It does not own model preprocessing, checkpoint loading,
training, logging, or scheduling.

The public boundary is intentionally small:

- Eval emits generic observations: `Mapping[str, numpy.ndarray | str | bool | int | float]`.
- Policy adapters consume those observations and return `numpy.ndarray` actions.
- Each benchmark driver documents the observation keys it emits and the action
  shape, dtype, range, and convention it expects.
- Callers can run policies in the same process or call a remote policy through
  `praxis-remote`.

This keeps benchmark logic inside `praxis-eval` and model-specific adaptation
inside the caller.

## Installation

Install the core package from PyPI:

```bash
pip install praxis-eval==0.1.1
```

The core package contains the public API, contracts, registry, result types,
setup/verify dispatchers, and local policy adapter. Install benchmark extras
for the simulator families you plan to run:

```bash
pip install "praxis-eval[remote]==0.1.1"
pip install "praxis-eval[libero]==0.1.1"
pip install "praxis-eval[metaworld]==0.1.1"
pip install "praxis-eval[robocasa]==0.1.1"
pip install "praxis-eval[robomimic]==0.1.1"
pip install "praxis-eval[simpler]==0.1.1"
pip install "praxis-eval[mshab]==0.1.1"
```

Use the source install when you need the current `main` branch before the next
release:

```bash
pip install "praxis-eval @ git+https://github.com/Chaoqi-LIU/praxis-eval.git@main"
```

`remote` installs the optional
[`praxis-remote`](https://github.com/Chaoqi-LIU/praxis-remote) adapter.
`libero`, `metaworld`, `robocasa`, and `robomimic` normally run inside the
current Python environment. `simpler` and `mshab` install lightweight package
wrappers into the current environment, but their simulator stacks usually run
best in dedicated runtimes created by `praxis-eval-setup`.

For local development:

```bash
git clone https://github.com/Chaoqi-LIU/praxis-eval.git
cd praxis-eval
uv sync --extra dev
uv run --extra dev pytest --strict-markers -m "not manual"
```

The repository has no simulator source submodules and no vendored
`third_party/` checkouts. Simulator forks are normal Python dependencies
published as `praxis-*` packages; setup commands download assets and create
runtime environments outside the source tree.

## Benchmark Extras

| Extra | Installs | Runtime model |
| --- | --- | --- |
| `remote` | `praxis-remote>=0.1.0,<0.2.0` | Optional gRPC policy client. |
| `libero` | `praxis-libero`, `praxis-robosuite` | Runs in the current Python environment. |
| `metaworld` | LeRobot Meta-World dependencies | Runs in the current Python environment. |
| `robocasa` | `praxis-robocasa`, `praxis-robosuite` | Runs in the current Python environment after asset setup. |
| `robomimic` | RoboMimic, `praxis-robosuite` | Runs in the current Python environment. |
| `simpler` | `praxis-simpler`, `praxis-remote` | Uses a dedicated SimplerEnv runtime for simulator execution. |
| `mshab` | `praxis-mshab`, `praxis-remote` | Uses a dedicated MS-HAB runtime for simulator execution. |

Prefer the narrowest extra that matches the benchmark you need. The `all` extra
is available for CI and broad integration environments, but it is usually not
the right choice for a policy-training environment because simulator dependency
sets can be heavy.

## Setup CLI

Setup commands prepare benchmark assets or dedicated simulator runtimes. Run the
setup command after installing the relevant extra.

Top-level help:

```bash
praxis-eval-setup --help
python -m praxis_eval.scripts.setup --help
```

RoboCasa setup downloads kitchen assets and writes `robocasa/macros_private.py`
with the dataset root used by RoboCasa:

```bash
pip install "praxis-eval[robocasa]==0.1.1"
praxis-eval-setup robocasa
praxis-eval-setup robocasa --dataset-base-path /data/robocasa
praxis-eval-setup robocasa --skip-download
```

SimplerEnv setup creates or updates a dedicated conda/micromamba environment
from the env spec shipped inside `praxis-simpler`, installs the matching
runtime packages into that environment, downloads ManiSkill assets, and writes
activation hooks for Vulkan/EGL and `MS_ASSET_DIR`:

```bash
pip install "praxis-eval[simpler]==0.1.1"
praxis-eval-setup simpler
praxis-eval-setup simpler --env-manager micromamba --env-name simpler-praxis
praxis-eval-setup simpler --ms-asset-dir /data/simpler/maniskill_assets
praxis-eval-setup simpler --skip-assets
```

MS-HAB setup follows the same model using the env spec shipped inside
`praxis-mshab`:

```bash
pip install "praxis-eval[mshab]==0.1.1"
praxis-eval-setup mshab
praxis-eval-setup mshab --env-manager micromamba --env-name mshab-praxis
praxis-eval-setup mshab --ms-asset-dir /data/mshab/maniskill_assets
praxis-eval-setup mshab --skip-assets
```

Each setup command supports `--help`:

```bash
praxis-eval-setup robocasa --help
praxis-eval-setup simpler --help
praxis-eval-setup mshab --help
```

## Verify CLI

Verifier commands run short random-action rollouts. They confirm that a
benchmark stack is importable, can create environments, can step, and can write
evaluation artifacts.

Top-level help:

```bash
praxis-eval-verify --help
python -m praxis_eval.scripts.verify --help
```

Current-environment verifiers:

```bash
praxis-eval-verify libero --task libero_10 --task-id 0
praxis-eval-verify metaworld --task reach-v3
praxis-eval-verify robocasa --task CloseToasterOvenDoor
praxis-eval-verify robomimic --task Lift --disable-render
```

Dedicated-runtime verifiers:

```bash
praxis-eval-verify simpler --env-name simpler-praxis --num-episodes 1 --num-envs 1
praxis-eval-verify mshab --env-name mshab-praxis --num-episodes 1 --num-envs 1
```

Pass an explicit interpreter when the runtime is a container, a manually managed
conda environment, or a cluster-provided Python binary:

```bash
praxis-eval-verify simpler --env-python-bin /path/to/simpler/bin/python
praxis-eval-verify mshab --env-python-bin /path/to/mshab/bin/python
```

Each verifier supports `--help`:

```bash
praxis-eval-verify libero --help
praxis-eval-verify metaworld --help
praxis-eval-verify robocasa --help
praxis-eval-verify robomimic --help
praxis-eval-verify simpler --help
praxis-eval-verify mshab --help
```

## Dedicated SimplerEnv and MS-HAB Runtimes

SimplerEnv and MS-HAB are special because their simulator dependencies are often
better isolated from the caller's main environment. The setup helpers create
dedicated conda or micromamba environments from env specs shipped by
`praxis-simpler` and `praxis-mshab`, then install `praxis-eval`,
`praxis-remote`, and the matching simulator runtime package into that dedicated
environment.

The setup helpers do not clone simulator source trees. They read runtime env
specs from installed package resources:

```text
praxis-simpler: simpler_env/praxis_conda_env.yaml
praxis-mshab:   mshab/praxis_conda_env.yaml
```

Default managed asset locations are:

```text
~/.cache/praxis_eval/assets/simpler/maniskill_assets
~/.cache/praxis_eval/assets/mshab/maniskill_assets
```

Override the asset location explicitly when a cluster or container already has a
fixed asset mount:

```bash
praxis-eval-setup simpler --ms-asset-dir /path/to/simpler/maniskill_assets
praxis-eval-setup mshab --ms-asset-dir /path/to/mshab/maniskill_assets
```

At evaluation time, pass the same path through the env config when you are not
using the default managed cache:

```python
from praxis_eval import EvalConfig

config = EvalConfig(
    task="widowx_carrot_on_plate",
    num_eval_per_task=5,
    output_dir="eval/simpler",
    env_kwargs={
        "python_bin": "/path/to/simpler-praxis/bin/python",
        "ms_asset_dir": "/data/simpler/maniskill_assets",
    },
)
```

The caller's policy environment does not need SimplerEnv or MS-HAB simulator
dependencies installed. It only needs `praxis-eval` and a policy adapter that
understands the observation/action contract.

## Supported Benchmarks

Built-in drivers are registered lazily when they are first requested.

| Driver | Extra | Notes |
| --- | --- | --- |
| `libero` | `libero` | LIBERO tasks through the Praxis-compatible fork. |
| `metaworld` | `metaworld` | Meta-World tasks through LeRobot integration. |
| `robocasa` | `robocasa` | RoboCasa tasks, assets, and metrics. |
| `robomimic` | `robomimic` | RoboMimic tasks and RoboSuite-backed rollouts. |
| `simpler` | `simpler` | Usually run through a dedicated interpreter via `env.python_bin`. |
| `mshab` | `mshab` | Usually run through a dedicated interpreter via `env.python_bin`. |

Check available registered drivers:

```python
from praxis_eval import available_drivers

print(available_drivers())
```

Inspect a benchmark contract:

```python
from praxis_eval import get_driver

driver = get_driver("libero")
print(driver.contract.observation_keys)
print(driver.contract.action)
```

## Evaluation API

Use `evaluate(env, policy, config)` for both local and remote policies:

```python
import numpy as np
from praxis_eval import EvalConfig, LocalPolicy, evaluate


def act(observations, *, policy_kwargs=None, episode_ids=None):
    del policy_kwargs, episode_ids
    return np.zeros((len(observations), 7), dtype=np.float32)


result = evaluate(
    "libero",
    policy=LocalPolicy(act),
    config=EvalConfig(
        task="libero_spatial",
        num_eval_per_task=10,
        num_parallel_env=4,
        output_dir="eval/libero_spatial",
    ),
)

print(result.overall)
```

`EvalConfig` contains generic rollout settings. Benchmark-specific settings go
in `env_kwargs`; policy-adapter settings go in `policy_kwargs`.

```python
from praxis_eval import EvalConfig

config = EvalConfig(
    task="libero_spatial",
    task_ids=(0, 1, 2),
    num_eval_per_task=5,
    num_parallel_env=3,
    output_dir="eval/libero_spatial",
    record_episodes_per_task=1,
    step_timeout_sec=30.0,
    env_kwargs={"max_episode_steps": 400},
    policy_kwargs={"temperature": 0.0},
)
```

The returned `EvalResult` has aggregate metrics, per-task metrics, optional
grouped metrics, artifact paths, and metadata:

```python
print(result.overall)
print(result.per_task)
print(result.artifacts)
```

## Observation and Action Contract

An observation is a flat mapping. Keys are explicit and namespaced so policy
adapters can stay generic:

```python
{
    "task": "put the mug on the plate",
    "observation.images.front": front_rgb,
    "observation.images.wrist": wrist_rgb,
    "observation.state": proprio,
    "metadata.episode_index": 7,
}
```

Values can be numpy arrays, strings, booleans, integers, floats, or numpy scalar
values. Arrays can represent RGB images, depth, state, masks, point clouds, or
calibration payloads. The benchmark contract tells the caller which keys are
present for that benchmark.

A policy adapter returns a batched `numpy.ndarray` action. `praxis-eval`
validates the action against the benchmark `ActionSpec` when a driver supplies
one. The policy itself does not need to know simulator internals; it only needs
to know the documented observation keys and action format.

## Remote Policies

Install the remote extra and run a policy server with `praxis-remote`:

```bash
pip install "praxis-eval[remote]==0.1.1"
```

Then evaluate through `RemotePolicy`:

```python
from praxis_eval import EvalConfig, RemotePolicy, evaluate

result = evaluate(
    "robocasa",
    policy=RemotePolicy("127.0.0.1:50051", timeout=30.0),
    config=EvalConfig(
        task="CloseDrawer",
        num_eval_per_task=10,
        output_dir="eval/robocasa",
    ),
)
```

Remote mode keeps the evaluator and policy in separate processes. This is useful
when the policy runtime and simulator runtime need incompatible dependencies.

## Repository Structure

```text
src/praxis_eval/
  api.py                  # evaluate(env, policy, config)
  contracts.py            # EnvContract, ObservationKey, ActionSpec
  registry.py             # driver registration and lookup
  types.py                # public EvalConfig, EvalResult, Policy protocol
  policies/               # local and remote policy adapters
  envs/                   # built-in benchmark drivers and env integration
  evaluation/             # rollout orchestration, metrics, artifacts
  scripts/                # setup and verify entry points
  managed_paths.py        # default cache paths for managed assets
tests/                    # package and integration tests
```

## Development Checks

Run the same checks as CI:

```bash
uv run --extra dev pre-commit run check-license-headers --all-files
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev pytest --strict-markers -m "not manual"
uv build --sdist --wheel
```

Contributor setup and project rules are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License and Attribution

`praxis-eval` is licensed under Apache-2.0. Redistribution must preserve the
license, copyright notices, and `NOTICE` file. If this package supports your
research or product work, please cite the repository using `CITATION.cff`.
