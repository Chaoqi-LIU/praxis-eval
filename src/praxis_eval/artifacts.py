# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluation artifact helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_artifact_dirs(output_dir: str | Path) -> tuple[Path, Path]:
    """Create and return ``(output_dir, media_dir)``."""
    root = Path(output_dir)
    media = root / "media"
    root.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)
    return root, media


def write_results_json(
    result: Any,
    output_dir: str | Path,
    *,
    filename: str = "results.json",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write an evaluation result payload to disk."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(result)
    if not isinstance(payload, dict):
        raise TypeError(f"result must serialize to a dict, got {type(payload)!r}")
    payload["_meta"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }
    path = root / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
