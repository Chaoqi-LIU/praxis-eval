#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Download the official RoboCasa GR-1 tabletop assets."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only report the installed GR-1 asset directory.",
    )
    return parser.parse_args()


def asset_root() -> Path:
    package_spec = importlib.util.find_spec("robocasa_gr1")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise ModuleNotFoundError(
            "RoboCasa GR-1 is not installed; install praxis-eval[robocasa_gr1]."
        )
    package_root = Path(next(iter(package_spec.submodule_search_locations)))
    return package_root / "models" / "assets"


def run_asset_download() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "robocasa_gr1.scripts.download_tabletop_assets",
            "--yes",
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    if args.skip_download:
        print("[setup_robocasa_gr1] Skipping tabletop asset download.")
    else:
        print("[setup_robocasa_gr1] Downloading tabletop assets...")
        run_asset_download()
    root = asset_root()
    print(f"[setup_robocasa_gr1] asset_root={root}")
    if not root.exists():
        raise FileNotFoundError(
            f"RoboCasa GR-1 asset directory does not exist after setup: {root}"
        )


if __name__ == "__main__":
    main()
