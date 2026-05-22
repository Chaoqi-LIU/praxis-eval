# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for praxis-eval verification scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def working_root() -> Path:
    """Return the caller's working tree for outputs and relative paths."""
    return Path.cwd().resolve()


def default_output_dir(name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return (working_root() / ".tmp" / "praxis_eval_verify" / name / timestamp).resolve()


def resolve_named_env_python(env_name: str) -> Path:
    probe = "import sys; print(sys.executable)"
    for manager in ("micromamba", "conda"):
        if shutil.which(manager) is None:
            continue
        result = subprocess.run(
            [manager, "run", "-n", env_name, "python", "-c", probe],
            cwd=str(working_root()),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        candidate = Path(result.stdout.strip()).expanduser().resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve python for env {env_name!r} via micromamba or conda."
    )


def resolve_env_python_bin(env_python_bin: str | Path | None, env_name: str) -> Path:
    if env_python_bin is not None:
        return Path(env_python_bin).expanduser().resolve()
    return resolve_named_env_python(env_name)


def resolve_output_dir(output_dir: str | Path | None, default_name: str) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return default_output_dir(default_name)


def run_command(
    command: list[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    argv = [str(part) for part in command]
    print(f"[verify] Running: {' '.join(argv)}")
    subprocess.run(
        argv,
        cwd=str(cwd or working_root()),
        env=env,
        check=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expected JSON object at {path}, got {type(payload).__name__}."
        )
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def validate_completed_episodes(*, verify_name: str, payload: dict[str, Any]) -> None:
    n_episodes = float(payload.get("n_episodes", 0.0))
    avg_episode_length = float(payload.get("avg_episode_length", 0.0))
    if n_episodes < 1.0:
        raise RuntimeError(
            f"{verify_name} produced no completed episodes: {n_episodes}."
        )
    if avg_episode_length <= 0.0:
        raise RuntimeError(
            f"{verify_name} produced a non-positive average episode length: "
            f"{avg_episode_length}."
        )


def print_summary(
    *, verify_name: str, results_path: Path, payload: dict[str, Any]
) -> None:
    success_rate = payload.get("success_rate")
    if success_rate is None:
        success_rate = payload.get("success_at_end_rate")
    if success_rate is None:
        success_rate = payload.get("success_once_rate")
    print(
        f"[{verify_name}] complete: "
        f"n_episodes={payload.get('n_episodes')} "
        f"success_rate={success_rate} "
        f"avg_episode_length={payload.get('avg_episode_length')} "
        f"results={results_path}"
    )


def env_or_default_path(env_name: str, default: Path) -> Path:
    raw = os.environ.get(env_name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default.expanduser().resolve()


def verify_imports(import_names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for import_name in import_names:
        module = __import__(import_name)
        versions[import_name] = str(getattr(module, "__version__", "unknown"))
    return versions


def python_module_command(module: str) -> list[str]:
    return [sys.executable, "-m", module]
