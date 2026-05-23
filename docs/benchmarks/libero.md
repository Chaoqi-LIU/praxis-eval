# LIBERO

LIBERO runs in the current Python environment through the `praxis-libero` and `praxis-robosuite` packages.

## Install

```bash
pip install "praxis-eval[libero]==0.1.1"
```

LIBERO config is written under `.tmp/libero_config` by the evaluator when needed. There is no `praxis-eval-setup libero` command.

## Verify

Run this only on a machine that can import LIBERO and run offscreen MuJoCo:

```bash
praxis-eval-verify libero --task libero_10 --task-id 0
praxis-eval-verify libero --help
```

## Task Selection

Default evaluator task: `libero_10`.

Known suite selectors include:

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`
- comma-separated suites such as `libero_spatial,libero_object`

Use `EvalConfig.task_ids` to select task ids inside each suite.

```python
from praxis_eval import EvalConfig

config = EvalConfig(
    task="libero_10",
    task_ids=(0, 1),
    num_eval_per_task=5,
    num_parallel_env=2,
    output_dir="eval/libero_10",
)
```

## Observation Format

Default config values:

| Option | Default |
| --- | --- |
| `obs_type` | `pixels_agent_pos` |
| `camera_name` | `agentview_image,robot0_eye_in_hand_image` |
| `camera_name_mapping` | `agentview_image -> image`, `robot0_eye_in_hand_image -> image2` |
| `observation_height` / `observation_width` | `360` / `360` |

Policy-facing keys:

| Key | Shape / dtype | Notes |
| --- | --- | --- |
| `task` | `str` | Natural-language LIBERO task instruction. |
| `observation.images.image` | `(3, H, W)`, `uint8` or `float32` | Default agent-view RGB camera after preprocessing. |
| `observation.images.image2` | `(3, H, W)`, `uint8` or `float32` | Default wrist/in-hand RGB camera after preprocessing. |
| `observation.state.eef_pos` | `(3,)` | Present with `obs_type="pixels_agent_pos"`. |
| `observation.state.eef_quat` | `(4,)` | Present with `obs_type="pixels_agent_pos"`. |
| `observation.state.eef_mat` | `(3, 3)` | Present with `obs_type="pixels_agent_pos"`. |
| `observation.state.gripper_qpos` | `(2,)` | Present with `obs_type="pixels_agent_pos"`. |
| `observation.state.gripper_qvel` | `(2,)` | Present with `obs_type="pixels_agent_pos"`. |
| `observation.state.joint_pos` | `(7,)` | Present with `obs_type="pixels_agent_pos"`. |
| `observation.state.joint_vel` | `(7,)` | Present with `obs_type="pixels_agent_pos"`. |

Set `obs_type="pixels"` to omit the robot-state fields.

## Action Format

| Field | Value |
| --- | --- |
| Shape | `(7,)` |
| Dtype | `float32` |
| Range | `[-1.0, 1.0]` |
| Convention | `normalized_delta_pose_gripper` |

The default control mode is `relative`. Actions are validated against the 7-D `ActionSpec` before stepping the environment.

## Runtime Notes

- LIBERO uses suite-specific maximum horizons when the suite supplies them.
- The evaluator can use initial states from the installed LIBERO package.
- Rendering is offscreen and depends on the local MuJoCo/OpenGL setup.

## Caveats

- LIBERO assets and init states must be available through the installed `praxis-libero` package.
- Custom cameras require a matching `camera_name_mapping`.
- The policy adapter is responsible for model-specific image normalization and tokenization.
