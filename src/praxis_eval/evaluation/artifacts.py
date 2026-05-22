# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for evaluation artifact paths and JSON serialization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_eval_artifact_paths(
    output_dir: str | Path,
    *,
    media_dirname: str = "media",
) -> tuple[Path, Path]:
    """Create and return ``(output_dir, media_dir)`` for eval artifacts."""
    out_dir = Path(output_dir)
    media_dir = out_dir / media_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, media_dir


def resolve_eval_step_dir(eval_root: str | Path, step: int) -> Path:
    """Return the canonical directory for per-step eval artifacts."""
    return Path(eval_root) / "by_step" / f"step_{int(step)}"


def write_eval_results_json(
    *,
    results: dict[str, Any],
    output_dir: str | Path,
    results_filename: str = "results.json",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write eval results payload to ``output_dir/results_filename``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = dict(results)
    payload["_meta"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }

    out_file = out_dir / results_filename
    out_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    return out_file
