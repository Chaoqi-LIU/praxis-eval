# SimplerEnv

SimplerEnv is evaluated through a dedicated runtime. The caller's policy environment does not need to import the simulator stack.

## Install

```bash
pip install "praxis-eval[simpler]"
```

This extra installs `praxis-simpler` and `praxis-remote` in the caller environment. The simulator itself usually runs in a dedicated conda or micromamba environment created by setup.

## Setup

```bash
praxis-eval-setup simpler
praxis-eval-setup simpler --env-manager micromamba --env-name simpler-praxis
praxis-eval-setup simpler --ms-asset-dir /data/simpler/maniskill_assets
praxis-eval-setup simpler --skip-assets
praxis-eval-setup simpler --help
```

Setup reads the env spec from the installed `praxis-simpler` package resource `simpler_env/praxis_conda_env.yaml`, installs `praxis-eval`, `praxis-remote`, and `praxis-simpler` into that runtime, downloads ManiSkill assets, and writes Vulkan/EGL activation hooks.

## Verify

Run this only on a machine with the dedicated SimplerEnv runtime:

```bash
praxis-eval-verify simpler --env-name simpler-praxis --num-episodes 1 --num-envs 1
praxis-eval-verify simpler --env-python-bin /path/to/simpler-praxis/bin/python
praxis-eval-verify simpler --help
```

## Task Selection

Default evaluator task: `bridge`.

| Selector | Environment id | Task id |
| --- | --- | --- |
| `widowx_carrot_on_plate` | `PutCarrotOnPlateInScene-v1` | `0` |
| `widowx_spoon_on_towel` | `PutSpoonOnTableClothInScene-v1` | `1` |
| `widowx_stack_cube` | `StackGreenCubeOnYellowCubeBakedTexInScene-v1` | `2` |
| `widowx_put_eggplant_in_basket` | `PutEggplantInBasketScene-v1` | `3` |
| `bridge` / `bridge_mt4` | all four tasks | all |

`EvalConfig.task_ids` filters the resolved task list by index.

## Observation Format

Policy-facing keys emitted by the SimplerEnv remote wrapper:

| Key | Shape / dtype | Notes |
| --- | --- | --- |
| `task` | `str` | Bridge task instruction. |
| `observation.images.image` | `(3, H, W)`, `uint8` or `float32` | Primary image key, configurable with `primary_image_key`. |
| `observation.state` | implementation-specific, often `(7,)`, `float32` | State key, configurable with `state_key`. |

The wrapper preserves `uint8` image transport when possible and converts HWC images to CHW.

## Action Format

| Field | Value |
| --- | --- |
| Shape | `(7,)` |
| Dtype | `float32` |
| Range | Not bounded by the public `ActionSpec` |
| Convention | `simpler_bridge_widowx_action` |

The external runtime formats policy actions into the SimplerEnv Bridge action dictionary. The first three values are the world vector, the next three are rotation delta, and the last value controls the gripper. `env_kwargs["action_scale"]` scales the action before it reaches the runtime adapter.

## Runtime Config

Pass the dedicated runtime interpreter through `env_kwargs`:

```python
from praxis_eval import EvalConfig

config = EvalConfig(
    task="widowx_carrot_on_plate",
    num_eval_per_task=5,
    num_parallel_env=2,
    output_dir="eval/simpler_carrot",
    env_kwargs={
        "python_bin": "/path/to/simpler-praxis/bin/python",
        "ms_asset_dir": "/data/simpler/maniskill_assets",
        "shader": "default",
    },
)
```

## Caveats

- Evaluation launches external subprocesses and communicates with the policy through `praxis-remote`.
- `env.python_bin` is required for normal evaluation unless the caller supplies an equivalent runtime path.
- Recording videos uses a dedicated single-env pass when enabled.
