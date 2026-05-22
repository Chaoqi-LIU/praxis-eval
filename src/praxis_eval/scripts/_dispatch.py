# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared command dispatcher for benchmark setup and verification CLIs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from importlib import import_module
from types import ModuleType


def dispatch_command(
    *,
    program: str,
    description: str,
    modules: Mapping[str, str],
    argv: Sequence[str] | None = None,
) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog=program,
        description=description,
        usage=f"{program} {{{','.join(sorted(modules))}}} ...",
    )
    parser.add_argument("benchmark", choices=sorted(modules))

    if not args:
        parser.print_help(sys.stderr)
        raise SystemExit(2)
    if args[0] in {"-h", "--help"}:
        parser.print_help()
        return

    namespace = parser.parse_args(args[:1])
    remaining = args[1:]
    module = import_module(modules[str(namespace.benchmark)])
    _run_module_main(module, program, str(namespace.benchmark), remaining)


def _run_module_main(
    module: ModuleType,
    program: str,
    benchmark: str,
    argv: Sequence[str],
) -> None:
    original_argv = sys.argv
    sys.argv = [f"{program} {benchmark}", *argv]
    try:
        module.main()
    finally:
        sys.argv = original_argv
