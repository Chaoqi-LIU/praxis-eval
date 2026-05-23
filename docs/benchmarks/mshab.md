# MS-HAB

MS-HAB is evaluated through a dedicated runtime. The caller's policy environment does not need to import the simulator stack.

## Install

```bash
pip install "praxis-eval[mshab]==0.1.1"
```

This extra installs `praxis-mshab` and `praxis-remote` in the caller environment. The simulator itself usually runs in a dedicated conda or micromamba environment created by setup.

## Setup

```bash
praxis-eval-setup mshab
praxis-eval-setup mshab --env-manager micromamba --env-name mshab-praxis
praxis-eval-setup mshab --ms-asset-dir /data/mshab/maniskill_assets
praxis-eval-setup mshab --skip-assets
praxis-eval-setup mshab --help
```

Setup reads the env spec from the installed `praxis-mshab` package resource `mshab/praxis_conda_env.yaml`, installs `praxis-eval`, `praxis-remote`, and `praxis-mshab` into that runtime, downloads ManiSkill assets, and writes Vulkan/EGL activation hooks.

## Verify

Run this only on a machine with the dedicated MS-HAB runtime:

```bash
praxis-eval-verify mshab --env-name mshab-praxis --num-episodes 1 --num-envs 1
praxis-eval-verify mshab --env-python-bin /path/to/mshab-praxis/bin/python
praxis-eval-verify mshab --help
```

## Task Selection

Default evaluator task: `set_table`.

| Selector | Subtask | Target | Task id |
| --- | --- | --- | --- |
| `pick` or `set_table_pick` | `pick` | `all` | `0` |
| `place` or `set_table_place` | `place` | `all` | `1` |
| `open_fridge` or `set_table_open_fridge` | `open` | `fridge` | `2` |
| `open_kitchen_counter` or `set_table_open_kitchen_counter` | `open` | `kitchen_counter` | `3` |
| `close_fridge` or `set_table_close_fridge` | `close` | `fridge` | `4` |
| `close_kitchen_counter` or `set_table_close_kitchen_counter` | `close` | `kitchen_counter` | `5` |
| `set_table` / `settable` | all six subtasks | mixed | all |

Dataset-name inference maps clean pick/place datasets to `pick,place` and full set-table datasets to `set_table`.

## Observation Format

Policy-facing keys emitted by the MS-HAB remote wrapper:

| Key | Shape / dtype | Notes |
| --- | --- | --- |
| `task` | `str` | Policy task string such as `set table: pick object`. |
| `observation.state` | `(42,)`, `float32` | State vector. |
| `observation.images.fetch_head` | `(3, 128, 128)`, `float32` | RGB image normalized to `[0, 1]`. |
| `observation.images.fetch_hand` | `(3, 128, 128)`, `float32` | RGB image normalized to `[0, 1]`. |

Depth images may exist in runtime observations, but the current policy-facing RGB contract filters out depth keys.

## Action Format

| Field | Value |
| --- | --- |
| Shape | `(13,)` |
| Dtype | `float32` |
| Range | `[-1.0, 1.0]` |
| Convention | `mshab_normalized_controller_action` |

The action is validated by `ActionSpec` before being sent through the remote policy path.

## Runtime Config

Pass the dedicated runtime interpreter through `env_kwargs`:

```python
from praxis_eval import EvalConfig

config = EvalConfig(
    task="set_table",
    num_eval_per_task=3,
    num_parallel_env=1,
    output_dir="eval/mshab_set_table",
    env_kwargs={
        "python_bin": "/path/to/mshab-praxis/bin/python",
        "ms_asset_dir": "/data/mshab/maniskill_assets",
        "split": "val",
        "obs_mode": "rgb",
    },
)
```

## Caveats

- Evaluation launches external subprocesses and communicates with the policy through `praxis-remote`.
- `env.python_bin` is required for normal evaluation unless the caller supplies an equivalent runtime path.
- Recording videos uses a dedicated single-env pass when enabled.
- The verifier defaults to a smaller random-action path; full evaluation uses `mshab.praxis_eval` from the runtime package.
