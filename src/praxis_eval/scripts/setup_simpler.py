#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap SIMPLER runtime setup for praxis-eval.

Steps:
1) Load the runtime env spec shipped by the installed `praxis-simpler` package.
2) Create/update the dedicated `simpler-praxis` conda/micromamba env from it.
3) Install `praxis-simpler`, `praxis-remote`, and `praxis-eval` into that env.
4) Download Bridge/WidowX ManiSkill assets into the managed praxis-eval cache.
5) Install the Vulkan/EGL activation hooks required on cluster GPU nodes.
6) Print an immediate import/status check from the resulting env.
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

SIMPLER_DISTRIBUTION = "praxis-simpler"

_ACTIVATE_HOOK_TEMPLATE = """#!/usr/bin/env bash

# Keep the Vulkan / EGL loader path explicit on cluster GPU nodes so SAPIEN and
# ManiSkill do not fall back to their bundled guesses.
export SIMPLER_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export SIMPLER_OLD_VK_ICD_FILENAMES="${VK_ICD_FILENAMES-}"
export SIMPLER_OLD_EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES-}"
export SIMPLER_OLD_VK_LAYER_PATH="${VK_LAYER_PATH-}"
export SIMPLER_OLD_XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR-}"
export SIMPLER_OLD_MS_ASSET_DIR="${MS_ASSET_DIR-}"

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
    export XDG_RUNTIME_DIR="/tmp/${USER:-$(id -un)}-simpler-xdg"
    mkdir -p "${XDG_RUNTIME_DIR}"
    chmod 700 "${XDG_RUNTIME_DIR}" 2>/dev/null || true
fi
"""

_DEACTIVATE_HOOK = """#!/usr/bin/env bash

if [ -n "${SIMPLER_OLD_LD_LIBRARY_PATH+x}" ]; then
    export LD_LIBRARY_PATH="${SIMPLER_OLD_LD_LIBRARY_PATH}"
    unset SIMPLER_OLD_LD_LIBRARY_PATH
else
    unset LD_LIBRARY_PATH
fi

if [ -n "${SIMPLER_OLD_VK_ICD_FILENAMES+x}" ]; then
    export VK_ICD_FILENAMES="${SIMPLER_OLD_VK_ICD_FILENAMES}"
    unset SIMPLER_OLD_VK_ICD_FILENAMES
else
    unset VK_ICD_FILENAMES
fi

if [ -n "${SIMPLER_OLD_EGL_VENDOR_LIBRARY_FILENAMES+x}" ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${SIMPLER_OLD_EGL_VENDOR_LIBRARY_FILENAMES}"
    unset SIMPLER_OLD_EGL_VENDOR_LIBRARY_FILENAMES
else
    unset __EGL_VENDOR_LIBRARY_FILENAMES
fi

if [ -n "${SIMPLER_OLD_VK_LAYER_PATH+x}" ]; then
    export VK_LAYER_PATH="${SIMPLER_OLD_VK_LAYER_PATH}"
    unset SIMPLER_OLD_VK_LAYER_PATH
else
    unset VK_LAYER_PATH
fi

if [ -n "${SIMPLER_OLD_XDG_RUNTIME_DIR+x}" ]; then
    export XDG_RUNTIME_DIR="${SIMPLER_OLD_XDG_RUNTIME_DIR}"
    unset SIMPLER_OLD_XDG_RUNTIME_DIR
else
    unset XDG_RUNTIME_DIR
fi

if [ -n "${SIMPLER_OLD_MS_ASSET_DIR+x}" ]; then
    export MS_ASSET_DIR="${SIMPLER_OLD_MS_ASSET_DIR}"
    unset SIMPLER_OLD_MS_ASSET_DIR
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
        default="simpler-praxis",
        help="Environment name to create/update.",
    )
    parser.add_argument(
        "--ms-asset-dir",
        default=None,
        help=(
            "ManiSkill asset root. Default: "
            "~/.cache/praxis_eval/assets/simpler/maniskill_assets."
        ),
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Skip downloading/refreshing the managed ManiSkill asset cache.",
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
    for asset_id in ("bridge_v2_real2sim", "widowx250s"):
        run_in_env(
            manager,
            env_name,
            cwd=cwd,
            args=[
                *env_prefix,
                "python",
                "-m",
                "mani_skill.utils.download_asset",
                "-y",
                asset_id,
            ],
        )


def print_asset_status(*, asset_root: Path) -> None:
    checks = {
        "bridge_v2_real2sim": asset_root
        / "data"
        / "tasks"
        / "bridge_v2_real2sim_dataset",
        "widowx250s": asset_root / "data" / "robots" / "widowx" / "wx250s.urdf",
    }
    print(f"[setup_simpler] ms_asset_dir={asset_root}")
    for name, path in checks.items():
        print(f"[setup_simpler] {name}={path.exists()} path={path}")


def print_status_check(
    *, manager: str, env_name: str, cwd: Path, ms_asset_dir: Path
) -> None:
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
                "import json, os, sys, torch, mani_skill, simpler_env, "
                "praxis_eval, praxis_remote; "
                "print(json.dumps({"
                "'python': sys.executable, "
                "'torch': getattr(torch, '__version__', 'unknown'), "
                "'mani_skill': getattr(mani_skill, '__version__', 'unknown'), "
                "'ms_asset_dir': os.environ.get('MS_ASSET_DIR'), "
                "'simpler_env': getattr(simpler_env, '__file__', 'unknown'), "
                "'praxis_eval': getattr(praxis_eval, '__file__', 'unknown'), "
                "'praxis_remote': getattr(praxis_remote, '__file__', 'unknown')"
                "}, indent=2, sort_keys=True))"
            ),
        ],
        capture_output=True,
    )
    print("[setup_simpler] Status check:")
    print(result.stdout.strip())


def main() -> None:
    args = parse_args()
    project = project_root()
    manager = resolve_env_manager(args.env_manager)
    cwd = project or Path.cwd().resolve()
    runtime_install = distribution_install_args(SIMPLER_DISTRIBUTION)
    env_file = distribution_resource_path(
        SIMPLER_DISTRIBUTION,
        "simpler_env/praxis_conda_env.yaml",
    )
    ms_asset_dir = (
        Path(args.ms_asset_dir).expanduser().resolve()
        if args.ms_asset_dir
        else managed_asset_dir("simpler")
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
    if not args.skip_assets:
        download_assets(
            manager=manager,
            env_name=args.env_name,
            cwd=cwd,
            ms_asset_dir=ms_asset_dir,
        )
    env_prefix = resolve_env_prefix(
        manager=manager,
        env_name=args.env_name,
        cwd=cwd,
    )
    install_activation_hooks(
        env_prefix,
        hook_name="simpler_vulkan.sh",
        activate_hook=_ACTIVATE_HOOK_TEMPLATE.replace(
            "__MS_ASSET_DIR__", ms_asset_dir.as_posix()
        ),
        deactivate_hook=_DEACTIVATE_HOOK,
    )

    print(f"[setup_simpler] manager={manager}")
    print(f"[setup_simpler] env_name={args.env_name}")
    print(f"[setup_simpler] env_prefix={env_prefix}")
    print(f"[setup_simpler] runtime_package={' '.join(runtime_install)}")
    print(f"[setup_simpler] conda_env={env_file}")
    print_asset_status(asset_root=ms_asset_dir)

    if not args.skip_status_check:
        print_status_check(
            manager=manager,
            env_name=args.env_name,
            cwd=cwd,
            ms_asset_dir=ms_asset_dir,
        )


if __name__ == "__main__":
    main()
