# RoboCasa GR-1

RoboCasa GR-1 is the official 24-task humanoid tabletop benchmark for the Fourier GR-1 arms, hands, and waist embodiment. It is a separate `robocasa_gr1` benchmark family from the RoboCasa365 `robocasa` driver.

## Install And Assets

```bash
pip install "praxis-eval[robocasa_gr1]"
praxis-eval-setup robocasa_gr1
```

The extra installs the PyPI packages `praxis-robocasa-gr1==0.2.0` and `praxis-robosuite==1.5.2.post1`. The setup command downloads the tabletop assets into the installed `robocasa_gr1` package. Run it in the same environment and shared filesystem used for evaluation.

Verify the installation on a simulator-capable GPU node:

```bash
praxis-eval-verify robocasa_gr1 \
  --task PnPCupToDrawerClose \
  --max-episode-steps 10
```

## Tasks

The default selector is `all`, containing the 24 tasks published by the upstream benchmark. The following selectors are available:

| Selector | Count | Contents |
| --- | ---: | --- |
| `all` or `gr1_24` | 24 | Complete official benchmark. |
| `articulated_6` | 6 | Placement tasks that finish by closing a drawer, microwave, or cabinet. |
| `rearrangement_18` | 18 | Post-training rearrangement tasks. |

A task may also be selected by its short name, generated environment class name, or full Gym id such as `gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env`.

## Policy Observation Contract

| Key | Shape / type |
| --- | --- |
| `video.ego_view_pad_res256_freq20` | `(3, 256, 256)`, `float32` in `[0, 1]` |
| `video.ego_view_bg_crop_pad_res256_freq20` | `(3, 256, 256)`, `float32` in `[0, 1]` |
| `state.left_arm`, `state.right_arm` | `(7,)`, `float32` |
| `state.left_hand`, `state.right_hand` | `(6,)`, `float32` |
| `state.waist` | `(3,)`, `float32` |
| `annotation.human.coarse_action` | language string |
| `task` | the same episode instruction for generic Praxis policies |

## Action Contract

Each environment step consumes one 29-D `float32` action in this order:

1. `action.left_arm` (7)
2. `action.right_arm` (7)
3. `action.left_hand` (6)
4. `action.right_hand` (6)
5. `action.waist` (3)

Values are absolute joint positions in physical units and are intentionally not clipped to `[-1, 1]`. A model may predict an action chunk, but its policy adapter must queue the chunk and return one 29-D step per evaluator call.

## Why It Runs In Process

The Praxis fork uses the independent Python namespace `robocasa_gr1`, while RoboCasa365 remains `robocasa`. Both use NumPy 2.2.5, MuJoCo 3.3.1, and `praxis-robosuite` 1.5.2.post1. Their robosuite environment registrations do not overlap, so both benchmark families can be installed and imported in one interpreter without a dedicated simulator subprocess.

Async evaluation still uses worker processes for parallel environments and OpenGL isolation. That is normal vectorized rollout execution, not a separate dependency runtime.
