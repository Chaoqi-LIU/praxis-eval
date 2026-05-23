# MetaWorld

MetaWorld runs in the current Python environment through LeRobot's MetaWorld dependency set.

## Install

```bash
pip install "praxis-eval[metaworld]==0.1.1"
```

There is no `praxis-eval-setup metaworld` command.

## Verify

Run this only on a machine that can run MetaWorld and offscreen MuJoCo:

```bash
praxis-eval-verify metaworld --task reach-v3
praxis-eval-verify metaworld --help
```

## Task Selection

Default evaluator task: `mt50`.

Selectors:

- `mt50` or `mt-50`
- difficulty groups: `easy`, `medium`, `hard`, `very_hard`
- a leaf task name from LeRobot's `metaworld_config.json`, such as `reach-v3`
- comma-separated selectors such as `easy,medium`

```python
from praxis_eval import EvalConfig

config = EvalConfig(
    task="reach-v3",
    num_eval_per_task=10,
    output_dir="eval/metaworld_reach",
)
```

## Observation Format

Default config values:

| Option | Default |
| --- | --- |
| `obs_type` | `pixels_agent_pos` |
| `camera_name` | `corner2` |
| `observation_height` / `observation_width` | `480` / `480` |
| `episode_length` | `500` |

Policy-facing keys:

| Key | Shape / dtype | Notes |
| --- | --- | --- |
| `task` | `str` | Task instruction from LeRobot MetaWorld metadata. |
| `observation.images.<camera>` | `(3, H, W)`, `uint8` or `float32` | Public contract. Default config maps MetaWorld pixels through LeRobot's image key. |
| `observation.state` | `(4,)`, `float32` or `float64` | First 4 elements of the raw MetaWorld observation when `obs_type="pixels_agent_pos"`. |

Set `obs_type="pixels"` to omit agent position state.

## Action Format

| Field | Value |
| --- | --- |
| Shape | `(4,)` |
| Dtype | `float32` |
| Range | `[-1.0, 1.0]` |
| Convention | `metaworld_xyz_gripper` |

Actions are validated against the normalized MetaWorld action space before stepping.

## Runtime Notes

- The wrapper lazily constructs the backend MetaWorld environment on first reset or render.
- The default `corner2` camera position is adjusted to match the LeRobot dataset convention.
- The wrapper flips `corner2` rendered images to match expected orientation.

## Caveats

- The exact list of leaf tasks comes from the installed LeRobot MetaWorld metadata.
- Rendering requires a working MuJoCo/OpenGL setup.
