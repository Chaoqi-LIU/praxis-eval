# MS-HAB Runtime

MS-HAB evaluation normally runs outside the caller's Python environment. `praxis-eval` starts a policy endpoint and launches the MS-HAB runtime as a subprocess.

## Command

```bash
praxis-eval-setup mshab
```

Useful options:

```bash
praxis-eval-setup mshab --env-manager micromamba --env-name mshab-praxis
praxis-eval-setup mshab --ms-asset-dir /data/mshab/maniskill_assets
praxis-eval-setup mshab --skip-assets
praxis-eval-setup mshab --skip-status-check
praxis-eval-setup mshab --help
```

## What It Does

The setup helper:

1. Locates the conda environment spec shipped by `praxis-mshab`: `mshab/praxis_conda_env.yaml`.
2. Creates or updates the requested conda or micromamba environment.
3. Installs `praxis-mshab`, `praxis-remote`, and `praxis-eval` into that environment.
4. Downloads ManiSkill assets: `ycb`, `ReplicaCAD`, and `ReplicaCADRearrange` unless `--skip-assets` is set.
5. Installs activation hooks for Vulkan/EGL and `MS_ASSET_DIR`.
6. Runs an import/status check unless `--skip-status-check` is set.

Default managed asset root:

```text
~/.cache/praxis_eval/assets/mshab/maniskill_assets
```

## Evaluation Config

Pass the runtime interpreter to evaluation:

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

## Verify

```bash
praxis-eval-verify mshab --env-name mshab-praxis --num-episodes 1 --num-envs 1
praxis-eval-verify mshab --env-python-bin /path/to/mshab-praxis/bin/python
```

## Caveats

- The simulator runtime must be able to import `mshab`, `praxis_eval`, and `praxis_remote`.
- The runtime expects ReplicaCAD rearrange task plans and spawn data below `MS_ASSET_DIR`.
- GPU cluster nodes often need explicit Vulkan/EGL variables; setup writes activation hooks for that reason.
