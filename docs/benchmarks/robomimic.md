# RoboMimic

RoboMimic runs robosuite tasks in the current Python environment.

## Install

```bash
pip install "praxis-eval[robomimic]"
```

There is no `praxis-eval-setup robomimic` command.

## Verify

Run this only on a machine that can run robosuite:

```bash
praxis-eval-verify robomimic --task Lift --disable-render
praxis-eval-verify robomimic --help
```

Remove `--disable-render` when verifying offscreen rendering:

```bash
praxis-eval-verify robomimic --task Lift --video-resolution 128
```

## Task Selection

Default evaluator task: `mt3`.

| Selector | Leaf tasks |
| --- | --- |
| `mt3` | `Lift`, `PickPlaceCan`, `NutAssemblySquare` |
| `mt4` | `Lift`, `PickPlaceCan`, `ToolHang`, `NutAssemblySquare` |

Leaf tasks and aliases are also supported:

- `Lift` or `lift`
- `PickPlaceCan`, `can`, or `pick_place_can`
- `NutAssemblySquare`, `square`, or `nut_assembly_square`
- `ToolHang` or `tool_hang`

## Observation Format

Default config values:

| Option | Default |
| --- | --- |
| `image_size` | `128` |
| `camera_names` | `agentview`, `robot0_eye_in_hand` |
| `state_ports` | `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos` |
| `video_camera` | `agentview` |
| `max_episode_steps` | `800`, overridden by known task horizons |
| `robot` | `Panda` |

Policy-facing keys:

| Key | Shape / dtype | Notes |
| --- | --- | --- |
| `task` | `str` | Natural-language instruction such as `Lift the cube.` |
| `observation.images.agentview` | `(3, H, W)`, usually `float32` | RGB camera after preprocessing. |
| `observation.images.robot0_eye_in_hand` | `(3, H, W)`, usually `float32` | RGB camera after preprocessing. |
| `observation.state` | `(9,)`, `float32` | Concatenated default state ports. |
| `observation.state.robot0_eef_pos` | `(3,)`, `float32` | Per-port alias retained by the processor. |
| `observation.state.robot0_eef_quat` | `(4,)`, `float32` | Per-port alias retained by the processor. |
| `observation.state.robot0_gripper_qpos` | `(2,)`, `float32` | Per-port alias retained by the processor. |

If you customize `state_ports`, the flat state dimension changes. Known state ports include `robot0_joint_pos`, `robot0_eef_pos`, `robot0_eef_quat`, and `robot0_gripper_qpos`.

## Action Format

| Field | Value |
| --- | --- |
| Shape | `(7,)` |
| Dtype | `float32` |
| Range | `[-1.0, 1.0]` |
| Convention | `robomimic_delta_pose_gripper` |

The wrapper validates shape, finite values, and action-space bounds before calling robosuite.

## Runtime Notes

- `MUJOCO_GL=egl` is set by the verifier when rendering is enabled.
- The wrapper uses the RoboMimic v1.5 PH-style robosuite controller setup.
- Known horizons are `Lift=100`, `PickPlaceCan=200`, `NutAssemblySquare=300`, and `ToolHang=800`.

## Caveats

- Rendering requires a working offscreen MuJoCo/OpenGL setup.
- The default state contract is narrower than all robosuite raw observations; customize `state_ports` if your policy needs additional proprioception.
