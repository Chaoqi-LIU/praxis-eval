# RoboCasa Assets

RoboCasa requires kitchen assets and a `macros_private.py` file in the installed RoboCasa package.

## Command

```bash
praxis-eval-setup robocasa
```

Useful options:

```bash
praxis-eval-setup robocasa --dataset-base-path /data/robocasa
praxis-eval-setup robocasa --skip-download
praxis-eval-setup robocasa --download-answer y
praxis-eval-setup robocasa --help
```

## What It Does

The setup helper:

1. Creates the dataset base directory.
2. Optionally runs `python -m robocasa.scripts.download_kitchen_assets`.
3. Copies `robocasa/macros.py` to `robocasa/macros_private.py` if needed.
4. Writes `DATASET_BASE_PATH = "<path>"` into `macros_private.py`.

Default dataset path resolution:

1. `$PRAXIS_EVAL_ROBOCASA_DATASET_ROOT`, when set.
2. `./data/robocasa` relative to the current working directory.

## When To Skip Download

Use `--skip-download` when assets are already installed or when a cluster image provides them:

```bash
praxis-eval-setup robocasa --skip-download --dataset-base-path /mnt/robocasa
```

The setup command still updates `DATASET_BASE_PATH`.

## Caveats

- The command modifies files inside the installed `robocasa` package.
- Run setup in the same environment that will run RoboCasa evaluation.
- Asset downloads can be large and may require network access accepted by the upstream downloader.
