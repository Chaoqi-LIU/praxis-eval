#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap MS-HAB runtime setup for praxis-eval.

This helper creates the dedicated MS-HAB conda/micromamba runtime from the env
spec shipped by the installed `praxis-mshab` package, installs the evaluation
packages into that runtime, and downloads ManiSkill assets into the managed
praxis-eval cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from praxis_eval.managed_paths import managed_asset_dir
from praxis_eval.scripts._conda_runtime import (
    create_or_update_env,
    install_activation_hooks,
    resolve_env_manager,
    resolve_env_prefix,
    run_in_env,
)
from praxis_eval.scripts._package_sources import (
    distribution_install_args,
    distribution_resource_path,
    praxis_eval_install,
    praxis_remote_install,
    project_root,
)

MSHAB_DISTRIBUTION = "praxis-mshab"

_ACTIVATE_HOOK = """#!/usr/bin/env bash

export MSHAB_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export MSHAB_OLD_VK_ICD_FILENAMES="${VK_ICD_FILENAMES-}"
export MSHAB_OLD_EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES-}"
export MSHAB_OLD_VK_LAYER_PATH="${VK_LAYER_PATH-}"
export MSHAB_OLD_XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR-}"
export MSHAB_OLD_MS_ASSET_DIR="${MS_ASSET_DIR-}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MS_ASSET_DIR="__MS_ASSET_DIR__"

if [ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
elif [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi

if [ -d "${CONDA_PREFIX}/share/vulkan/explicit_layer.d" ]; then
    export VK_LAYER_PATH="${CONDA_PREFIX}/share/vulkan/explicit_layer.d${VK_LAYER_PATH:+:${VK_LAYER_PATH}}"
fi

if [ -z "${XDG_RUNTIME_DIR:-}" ]; then
    export XDG_RUNTIME_DIR="/tmp/${USER:-$(id -un)}-mshab-xdg"
    mkdir -p "${XDG_RUNTIME_DIR}"
    chmod 700 "${XDG_RUNTIME_DIR}" 2>/dev/null || true
fi
"""

_DEACTIVATE_HOOK = """#!/usr/bin/env bash

if [ -n "${MSHAB_OLD_LD_LIBRARY_PATH+x}" ]; then
    export LD_LIBRARY_PATH="${MSHAB_OLD_LD_LIBRARY_PATH}"
    unset MSHAB_OLD_LD_LIBRARY_PATH
else
    unset LD_LIBRARY_PATH
fi

if [ -n "${MSHAB_OLD_VK_ICD_FILENAMES+x}" ]; then
    export VK_ICD_FILENAMES="${MSHAB_OLD_VK_ICD_FILENAMES}"
    unset MSHAB_OLD_VK_ICD_FILENAMES
else
    unset VK_ICD_FILENAMES
fi

if [ -n "${MSHAB_OLD_EGL_VENDOR_LIBRARY_FILENAMES+x}" ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${MSHAB_OLD_EGL_VENDOR_LIBRARY_FILENAMES}"
    unset MSHAB_OLD_EGL_VENDOR_LIBRARY_FILENAMES
else
    unset __EGL_VENDOR_LIBRARY_FILENAMES
fi

if [ -n "${MSHAB_OLD_VK_LAYER_PATH+x}" ]; then
    export VK_LAYER_PATH="${MSHAB_OLD_VK_LAYER_PATH}"
    unset MSHAB_OLD_VK_LAYER_PATH
else
    unset VK_LAYER_PATH
fi

if [ -n "${MSHAB_OLD_XDG_RUNTIME_DIR+x}" ]; then
    export XDG_RUNTIME_DIR="${MSHAB_OLD_XDG_RUNTIME_DIR}"
    unset MSHAB_OLD_XDG_RUNTIME_DIR
else
    unset XDG_RUNTIME_DIR
fi

if [ -n "${MSHAB_OLD_MS_ASSET_DIR+x}" ]; then
    export MS_ASSET_DIR="${MSHAB_OLD_MS_ASSET_DIR}"
    unset MSHAB_OLD_MS_ASSET_DIR
else
    unset MS_ASSET_DIR
fi
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-manager",
        choices=("auto", "micromamba", "conda"),
        default="auto",
        help="Environment manager to use. Default: auto-detect micromamba, then conda.",
    )
    parser.add_argument(
        "--env-name",
        default="mshab-praxis",
        help="Environment name to create/update.",
    )
    parser.add_argument(
        "--ms-asset-dir",
        default=None,
        help=(
            "ManiSkill asset root. Default: "
            "~/.cache/praxis_eval/assets/mshab/maniskill_assets."
        ),
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Skip downloading/refreshing the MS-HAB asset cache.",
    )
    parser.add_argument(
        "--skip-status-check",
        action="store_true",
        help="Skip the final import/status check.",
    )
    return parser.parse_args()


def download_assets(
    *,
    manager: str,
    env_name: str,
    cwd: Path,
    ms_asset_dir: Path,
) -> None:
    ms_asset_dir = ms_asset_dir.expanduser().resolve()
    ms_asset_dir.mkdir(parents=True, exist_ok=True)
    (ms_asset_dir / "data").mkdir(parents=True, exist_ok=True)
    env_prefix = [
        "env",
        f"MS_ASSET_DIR={ms_asset_dir}",
    ]
    for dataset in ("ycb", "ReplicaCAD", "ReplicaCADRearrange"):
        run_in_env(
            manager,
            env_name,
            cwd=cwd,
            args=[
                *env_prefix,
                "python",
                "-m",
                "mani_skill.utils.download_asset",
                dataset,
            ],
        )


def print_asset_status(*, asset_root: Path) -> None:
    rearrange_root = (
        asset_root / "data" / "scene_datasets" / "replica_cad_dataset" / "rearrange"
    )
    checks = {
        "pick_all": rearrange_root
        / "task_plans"
        / "set_table"
        / "pick"
        / "val"
        / "all.json",
        "open_fridge": rearrange_root
        / "task_plans"
        / "set_table"
        / "open"
        / "val"
        / "fridge.json",
        "spawn_pick": rearrange_root
        / "spawn_data"
        / "set_table"
        / "pick"
        / "val"
        / "spawn_data.pt",
    }
    print(f"[setup_mshab] ms_asset_dir={asset_root}")
    for name, path in checks.items():
        print(f"[setup_mshab] {name}={path.exists()} path={path}")


def print_status(*, manager: str, env_name: str, cwd: Path, ms_asset_dir: Path) -> None:
    result = run_in_env(
        manager,
        env_name,
        cwd=cwd,
        args=[
            "env",
            f"MS_ASSET_DIR={ms_asset_dir}",
            "python",
            "-c",
            (
                "import json, os, mani_skill, mshab, praxis_eval, praxis_remote, torch; "
                "print('python ok'); "
                "print(json.dumps({"
                "'mani_skill': mani_skill.__file__, "
                "'mshab': mshab.__file__, "
                "'ms_asset_dir': os.environ.get('MS_ASSET_DIR'), "
                "'praxis_eval': praxis_eval.__file__, "
                "'praxis_remote': praxis_remote.__file__, "
                "'torch': torch.__version__"
                "}, sort_keys=True))"
            ),
        ],
        capture_output=True,
    )
    print("[setup_mshab] Status check:")
    print(result.stdout.strip())


def main() -> None:
    args = parse_args()
    project = project_root()
    manager = resolve_env_manager(args.env_manager)
    cwd = project or Path.cwd().resolve()
    runtime_install = distribution_install_args(MSHAB_DISTRIBUTION)
    env_file = distribution_resource_path(
        MSHAB_DISTRIBUTION,
        "mshab/praxis_conda_env.yaml",
    )

    create_or_update_env(
        manager=manager,
        env_name=args.env_name,
        env_file=env_file,
        cwd=cwd,
    )
    run_in_env(
        manager,
        args.env_name,
        cwd=cwd,
        args=[
            "python",
            "-m",
            "pip",
            "install",
            *runtime_install,
            *praxis_remote_install(project),
            *praxis_eval_install(project),
        ],
    )
    asset_root = (
        Path(args.ms_asset_dir).expanduser().resolve()
        if args.ms_asset_dir
        else managed_asset_dir("mshab")
    )
    if not args.skip_assets:
        download_assets(
            manager=manager,
            env_name=args.env_name,
            cwd=cwd,
            ms_asset_dir=asset_root,
        )
    env_prefix = resolve_env_prefix(
        manager=manager,
        env_name=args.env_name,
        cwd=cwd,
    )
    install_activation_hooks(
        env_prefix,
        hook_name="mshab_praxis_runtime.sh",
        activate_hook=_ACTIVATE_HOOK.replace("__MS_ASSET_DIR__", asset_root.as_posix()),
        deactivate_hook=_DEACTIVATE_HOOK,
    )

    print(f"[setup_mshab] manager={manager}")
    print(f"[setup_mshab] env_name={args.env_name}")
    print(f"[setup_mshab] env_prefix={env_prefix}")
    print(f"[setup_mshab] runtime_package={' '.join(runtime_install)}")
    print(f"[setup_mshab] conda_env={env_file}")
    print_asset_status(asset_root=asset_root)

    if not args.skip_status_check:
        print_status(
            manager=manager,
            env_name=args.env_name,
            cwd=cwd,
            ms_asset_dir=asset_root,
        )


if __name__ == "__main__":
    main()
