# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Install sources used by setup helpers for nested runtime envs."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from urllib.parse import unquote, urlparse

PRAXIS_REMOTE_FALLBACK_INSTALL = "praxis-remote>=0.1.0,<0.2.0"

PRAXIS_EVAL_FALLBACK_INSTALL = "praxis-eval==0.1.3"


def project_root() -> Path | None:
    """Return the source checkout root when running from a praxis-eval checkout."""
    root = Path(__file__).resolve().parents[3]
    if (root / "pyproject.toml").exists() and (root / "src" / "praxis_eval").exists():
        return root.resolve()
    return None


def distribution_resource_path(distribution_name: str, relative_path: str) -> Path:
    """Return a file shipped by an installed distribution without importing it."""
    try:
        dist = distribution(distribution_name)
    except PackageNotFoundError as exc:
        raise PackageNotFoundError(
            f"Missing required distribution {distribution_name!r}. Install the "
            "matching praxis-eval extra before running this setup helper."
        ) from exc

    path = Path(dist.locate_file(relative_path)).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{distribution_name!r} does not provide required resource "
            f"{relative_path!r}. Upgrade the package to the version required by "
            "praxis-eval."
        )
    return path


def distribution_install_args(
    distribution_name: str,
    *,
    fallback: str | None = None,
) -> list[str]:
    """Return pip install args that preserve an installed distribution's source."""
    try:
        dist = distribution(distribution_name)
    except PackageNotFoundError:
        if fallback is not None:
            return [fallback]
        raise

    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text:
        install_args = _direct_url_install_args(distribution_name, direct_url_text)
        if install_args is not None:
            return install_args

    return [f"{distribution_name}=={dist.version}"]


def praxis_eval_install(root: Path | None) -> list[str]:
    if root is not None:
        return ["-e", str(root)]
    return distribution_install_args(
        "praxis-eval",
        fallback=PRAXIS_EVAL_FALLBACK_INSTALL,
    )


def praxis_remote_install(root: Path | None) -> list[str]:
    sibling = root.parent / "praxis-remote" if root is not None else None
    if (
        sibling is not None
        and (sibling / "pyproject.toml").exists()
        and (sibling / "src" / "praxis_remote").exists()
    ):
        return ["-e", str(sibling)]
    return distribution_install_args(
        "praxis-remote",
        fallback=PRAXIS_REMOTE_FALLBACK_INSTALL,
    )


def _direct_url_install_args(
    distribution_name: str,
    direct_url_text: str,
) -> list[str] | None:
    data = json.loads(direct_url_text)
    url = data.get("url")
    if not isinstance(url, str) or not url:
        return None

    dir_info = data.get("dir_info")
    editable = isinstance(dir_info, dict) and dir_info.get("editable") is True
    if editable:
        path = _file_url_path(url)
        if path is not None:
            return ["-e", str(path)]

    vcs_info = data.get("vcs_info")
    if isinstance(vcs_info, dict):
        vcs = vcs_info.get("vcs")
        revision = vcs_info.get("commit_id") or vcs_info.get("requested_revision")
        if isinstance(vcs, str) and vcs and isinstance(revision, str) and revision:
            direct_url = url if url.startswith(f"{vcs}+") else f"{vcs}+{url}"
            if editable:
                return ["-e", f"{direct_url}@{revision}#egg={distribution_name}"]
            return [f"{distribution_name} @ {direct_url}@{revision}"]
        return None

    return [f"{distribution_name} @ {url}"]


def _file_url_path(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve()
