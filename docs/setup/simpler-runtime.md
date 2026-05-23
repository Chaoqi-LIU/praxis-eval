# SimplerEnv Runtime

SimplerEnv evaluation normally runs outside the caller's Python environment. `praxis-eval` starts a policy endpoint and launches the SimplerEnv runtime as a subprocess.

## Command

```bash
praxis-eval-setup simpler
```

Useful options:

```bash
praxis-eval-setup simpler --env-manager micromamba --env-name simpler-praxis
praxis-eval-setup simpler --ms-asset-dir /data/simpler/maniskill_assets
praxis-eval-setup simpler --skip-assets
praxis-eval-setup simpler --skip-status-check
praxis-eval-setup simpler --help
```

## What It Does

The setup helper:

1. Locates the conda environment spec shipped by `praxis-simpler`: `simpler_env/praxis_conda_env.yaml`.
2. Creates or updates the requested conda or micromamba environment.
3. Installs `praxis-simpler`, `praxis-remote`, and `praxis-eval` into that environment.
4. Downloads the `bridge_v2_real2sim` and `widowx250s` ManiSkill assets unless `--skip-assets` is set.
5. Installs activation hooks for Vulkan/EGL and `MS_ASSET_DIR`.
6. Runs an import/status check unless `--skip-status-check` is set.

Default managed asset root:

```text
~/.cache/praxis_eval/assets/simpler/maniskill_assets
```

## Evaluation Config

Pass the runtime interpreter to evaluation:

```python
from praxis_eval import EvalConfig

config = EvalConfig(
    task="bridge",
    num_eval_per_task=4,
    num_parallel_env=2,
    output_dir="eval/simpler_bridge",
    env_kwargs={
        "python_bin": "/path/to/simpler-praxis/bin/python",
        "ms_asset_dir": "/data/simpler/maniskill_assets",
        "shader": "default",
    },
)
```

## Verify

```bash
praxis-eval-verify simpler --env-name simpler-praxis --num-episodes 1 --num-envs 1
praxis-eval-verify simpler --env-python-bin /path/to/simpler-praxis/bin/python
```

## Caveats

- The caller environment only needs `praxis-eval`, policy dependencies, and the optional remote transport path.
- The simulator runtime must be able to import `simpler_env`, `praxis_eval`, and `praxis_remote`.
- GPU cluster nodes often need explicit Vulkan/EGL variables; setup writes activation hooks for that reason.
