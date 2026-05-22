#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap RoboCasa runtime setup for praxis-eval.

Steps:
1) Download RoboCasa kitchen assets (optional, enabled by default)
2) Ensure robocasa/macros_private.py exists
3) Set DATASET_BASE_PATH in macros_private.py

Usage:
    uv run python -m praxis_eval.scripts.setup_robocasa
    uv run python -m praxis_eval.scripts.setup_robocasa --skip-download
    uv run python -m praxis_eval.scripts.setup_robocasa --dataset-base-path /abs/path
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip `python -m robocasa.scripts.download_kitchen_assets`.",
    )
    parser.add_argument(
        "--dataset-base-path",
        default=None,
        help=(
            "Override DATASET_BASE_PATH. Default: "
            "$PRAXIS_EVAL_ROBOCASA_DATASET_ROOT or ./data/robocasa"
        ),
    )
    parser.add_argument(
        "--download-answer",
        default="y",
        choices=("y", "n"),
        help="Answer sent to the kitchen-assets downloader prompt (default: y).",
    )
    return parser.parse_args()


def default_dataset_base_path() -> Path:
    configured = os.environ.get("PRAXIS_EVAL_ROBOCASA_DATASET_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / "data" / "robocasa").resolve()


def run_kitchen_asset_download(*, answer: str) -> None:
    cmd = [sys.executable, "-m", "robocasa.scripts.download_kitchen_assets"]
    subprocess.run(
        cmd,
        input=f"{answer}\n",
        text=True,
        check=True,
    )


def ensure_macros_private() -> tuple[Path, Path]:
    import robocasa

    pkg_root = Path(robocasa.__path__[0])
    macros = pkg_root / "macros.py"
    macros_private = pkg_root / "macros_private.py"

    if not macros.exists():
        raise FileNotFoundError(f"Missing robocasa macros.py at {macros}")

    if not macros_private.exists():
        shutil.copyfile(macros, macros_private)

    return macros, macros_private


def set_dataset_base_path(macros_private: Path, dataset_base_path: Path) -> None:
    line = f'DATASET_BASE_PATH = "{dataset_base_path.as_posix()}"'
    lines = macros_private.read_text().splitlines()
    kept = [entry for entry in lines if not entry.startswith("DATASET_BASE_PATH")]
    kept.append(line)
    macros_private.write_text("\n".join(kept) + "\n")


def main() -> None:
    args = parse_args()

    dataset_base_path = (
        Path(args.dataset_base_path).expanduser().resolve()
        if args.dataset_base_path is not None
        else default_dataset_base_path()
    )
    dataset_base_path.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("[setup_robocasa] Downloading RoboCasa kitchen assets...")
        run_kitchen_asset_download(answer=args.download_answer)
    else:
        print("[setup_robocasa] Skipping kitchen asset download.")

    _, macros_private = ensure_macros_private()
    set_dataset_base_path(macros_private, dataset_base_path)

    print(f"[setup_robocasa] Updated {macros_private}")
    print(f"[setup_robocasa] DATASET_BASE_PATH={dataset_base_path}")


if __name__ == "__main__":
    main()
