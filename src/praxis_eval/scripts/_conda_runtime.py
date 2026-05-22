# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared conda/micromamba helpers for special simulator runtimes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run(
    cmd: list[str], *, cwd: Path, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run one setup command in a source checkout."""
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        check=True,
        capture_output=capture_output,
    )


def resolve_env_manager(name: str) -> str:
    """Resolve the requested conda-compatible environment manager."""
    if name != "auto":
        resolved = shutil.which(name)
        if resolved is None:
            raise FileNotFoundError(f"Requested env manager {name!r} is not on PATH.")
        return resolved

    for candidate in ("micromamba", "conda"):
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    raise FileNotFoundError("Could not find either `micromamba` or `conda` on PATH.")


def env_exists(manager: str, env_name: str, *, cwd: Path) -> bool:
    """Return whether a named conda-compatible environment can run Python."""
    result = subprocess.run(
        [
            manager,
            "run",
            "-n",
            env_name,
            "python",
            "-c",
            "import sys; print(sys.prefix)",
        ],
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def create_or_update_env(
    *,
    manager: str,
    env_name: str,
    env_file: Path,
    cwd: Path,
) -> None:
    """Create or update a named conda-compatible runtime from an env YAML."""
    action = "update" if env_exists(manager, env_name, cwd=cwd) else "create"
    command = [
        manager,
        "env",
        action,
        "-y",
        "--name",
        env_name,
        "--file",
        str(env_file),
    ]
    if action == "update":
        command.append("--prune")
    run(command, cwd=cwd)


def run_in_env(
    manager: str,
    env_name: str,
    *,
    cwd: Path,
    args: list[str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one command inside a named conda-compatible environment."""
    return run(
        [manager, "run", "-n", env_name, *args],
        cwd=cwd,
        capture_output=capture_output,
    )


def resolve_env_prefix(*, manager: str, env_name: str, cwd: Path) -> Path:
    """Resolve ``sys.prefix`` for a named runtime environment."""
    result = run_in_env(
        manager,
        env_name,
        cwd=cwd,
        args=["python", "-c", "import sys; print(sys.prefix)"],
        capture_output=True,
    )
    return Path(result.stdout.strip()).resolve()


def install_activation_hooks(
    env_prefix: Path,
    *,
    hook_name: str,
    activate_hook: str,
    deactivate_hook: str,
) -> None:
    """Install conda activation/deactivation hook files."""
    activate_dir = env_prefix / "etc" / "conda" / "activate.d"
    deactivate_dir = env_prefix / "etc" / "conda" / "deactivate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)
    deactivate_dir.mkdir(parents=True, exist_ok=True)

    activate_path = activate_dir / hook_name
    deactivate_path = deactivate_dir / hook_name
    activate_path.write_text(activate_hook, encoding="utf-8")
    deactivate_path.write_text(deactivate_hook, encoding="utf-8")
    activate_path.chmod(0o755)
    deactivate_path.chmod(0o755)
