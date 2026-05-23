# RoboCasa

RoboCasa runs in the current Python environment and requires RoboCasa kitchen assets.

## Install

```bash
pip install "praxis-eval[robocasa]==0.1.1"
```

## Setup

```bash
praxis-eval-setup robocasa
praxis-eval-setup robocasa --dataset-base-path /data/robocasa
praxis-eval-setup robocasa --skip-download
praxis-eval-setup robocasa --help
```

The setup command downloads kitchen assets unless `--skip-download` is passed, creates `robocasa/macros_private.py` when needed, and writes `DATASET_BASE_PATH`. The default dataset root is `$PRAXIS_EVAL_ROBOCASA_DATASET_ROOT` or `./data/robocasa`.

## Verify

Run this only on a machine that can run RoboCasa:

```bash
praxis-eval-verify robocasa --task CloseToasterOvenDoor
praxis-eval-verify robocasa --help
```

## Task Selection

Default evaluator task: `mt5`.

Built-in evaluator group:

| Selector | Leaf tasks |
| --- | --- |
| `mt5` | `CloseToasterOvenDoor`, `OpenDrawer`, `PickPlaceDrawerToCounter`, `TurnOnElectricKettle`, `SlideDishwasherRack` |

You may also pass a RoboCasa365 leaf task such as `CloseToasterOvenDoor` or a task set advertised by the installed RoboCasa registry.

```python
from praxis_eval import EvalConfig

config = EvalConfig(
    task="mt5",
    num_eval_per_task=5,
    num_parallel_env=2,
    output_dir="eval/robocasa_mt5",
)
```

## Observation Format

Default config values:

| Option | Default |
| --- | --- |
| `image_size` | `128` |
| `camera_names` | `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand` |
| `split` | `all` |
| `max_episode_steps` | `500`, overridden by task-specific horizons when known |

Policy-facing keys:

| Key | Shape / dtype | Notes |
| --- | --- | --- |
| `task` | `str` | Episode language instruction from RoboCasa metadata. |
| `observation.images.robot0_agentview_left` | `(3, H, W)`, usually `float32` | RGB camera after LeRobot preprocessing. |
| `observation.images.robot0_agentview_right` | `(3, H, W)`, usually `float32` | RGB camera after LeRobot preprocessing. |
| `observation.images.robot0_eye_in_hand` | `(3, H, W)`, usually `float32` | RGB camera after LeRobot preprocessing. |
| `observation.state` | `(16,)`, `float32` | Concatenated official RoboCasa v1.0 state modality. |
| `observation.state.robot0_base_pos` | `(3,)`, `float32` | Per-key alias retained by the processor. |
| `observation.state.robot0_base_quat` | `(4,)`, `float32` | Per-key alias retained by the processor. |
| `observation.state.robot0_base_to_eef_pos` | `(3,)`, `float32` | Per-key alias retained by the processor. |
| `observation.state.robot0_base_to_eef_quat` | `(4,)`, `float32` | Per-key alias retained by the processor. |
| `observation.state.robot0_gripper_qpos` | `(2,)`, `float32` | Per-key alias retained by the processor. |

## Action Format

| Field | Value |
| --- | --- |
| Shape | `(12,)` |
| Dtype | `float32` |
| Range | `[-1.0, 1.0]` at the public evaluator contract |
| Convention | `robocasa_normalized_mobile_manipulator_action` |

The evaluator expects the official RoboCasa365 LeRobot action order: base motion, control mode, end-effector position, end-effector rotation, and gripper close. The wrapper maps that order to RoboCasa's native controller order before stepping the simulator and clips to task-specific native controller bounds when needed.

## Runtime Notes

- RoboCasa uses fork-based async workers when available.
- Environment construction retries known intermittent MuJoCo layout failures with shifted seeds.
- The wrapper flips RoboCasa camera images to correct robosuite's image orientation.

## Caveats

- Asset setup mutates the installed RoboCasa package by writing `macros_private.py`.
- Large Objaverse visual meshes may be neutralized by a runtime guard to avoid MuJoCo XML failures.
- Some scenes are expensive to build; the eval pool has a long build timeout for retargeting.
