# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import stat
import sys
from pathlib import Path

from praxis_eval.envs.subprocess_runtime import package_root_for_python


def _python_with_path(tmp_path: Path, package_parent: Path) -> Path:
    python_bin = tmp_path / "python-with-path"
    python_bin.write_text(
        f'#!/usr/bin/env sh\nPYTHONPATH={package_parent} exec {sys.executable} "$@"\n',
        encoding="utf-8",
    )
    python_bin.chmod(python_bin.stat().st_mode | stat.S_IXUSR)
    return python_bin


def test_package_root_for_python_supports_namespace_packages(tmp_path: Path) -> None:
    package_parent = tmp_path / "site"
    namespace_root = package_parent / "namespace_pkg"
    namespace_root.mkdir(parents=True)
    (namespace_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    python_bin = _python_with_path(tmp_path, package_parent)

    assert package_root_for_python(python_bin, "namespace_pkg") == namespace_root


def test_package_root_for_python_supports_regular_packages(tmp_path: Path) -> None:
    package_parent = tmp_path / "site"
    package_root = package_parent / "regular_pkg"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    python_bin = _python_with_path(tmp_path, package_parent)

    assert package_root_for_python(python_bin, "regular_pkg") == package_root
