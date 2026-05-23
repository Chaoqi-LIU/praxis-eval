#!/usr/bin/env python3
"""Cheap documentation checks that do not run simulator benchmarks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def markdown_paths() -> list[Path]:
    paths = [ROOT / "README.md"]
    paths.extend((ROOT / "docs").rglob("*.md"))
    return [
        path
        for path in sorted(paths)
        if "node_modules" not in path.parts
        and ".vitepress" not in path.parts
        and path.is_file()
    ]


def check_python_fences() -> None:
    failures: list[str] = []
    fence_count = 0

    for path in markdown_paths():
        rel = path.relative_to(ROOT)
        in_block = False
        language = ""
        start_line = 0
        buffer: list[str] = []

        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if line.startswith("```"):
                if not in_block:
                    in_block = True
                    language = line[3:].strip().split()[0] if line[3:].strip() else ""
                    start_line = line_number + 1
                    buffer = []
                    continue

                if language == "python":
                    fence_count += 1
                    code = "\n".join(buffer) + "\n"
                    try:
                        compile(code, f"{rel}:{start_line}", "exec")
                    except SyntaxError as exc:
                        failures.append(f"{rel}:{start_line}: {exc.msg}")

                in_block = False
                language = ""
                start_line = 0
                buffer = []
                continue

            if in_block:
                buffer.append(line)

        if in_block:
            failures.append(f"{rel}:{start_line}: unclosed fenced code block")

    if failures:
        print("Python fence syntax failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Compiled {fence_count} Python fenced code blocks.")


def run_python_module(args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SRC)
        if not env.get("PYTHONPATH")
        else f"{SRC}{os.pathsep}{env['PYTHONPATH']}"
    )
    command = [sys.executable, "-m", *args]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    label = " ".join(args)
    if proc.returncode != 0:
        print(f"Command failed: python -m {label}", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    print(f"Checked python -m {label}")


def check_public_api_smoke() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from praxis_eval import (  # noqa: PLC0415
        ActionSpec,
        EnvContract,
        EvalConfig,
        LocalPolicy,
        RemotePolicy,
        available_drivers,
        get_driver,
    )

    del ActionSpec, EnvContract, EvalConfig, LocalPolicy, RemotePolicy
    expected = {"libero", "metaworld", "mshab", "robocasa", "robomimic", "simpler"}
    actual = set(available_drivers())
    missing = expected - actual
    if missing:
        raise SystemExit(f"Missing built-in drivers: {sorted(missing)}")

    for name in sorted(expected):
        contract = get_driver(name).contract
        if contract.env_type != name:
            raise SystemExit(f"{name} contract env_type is {contract.env_type!r}")

    print("Checked public API imports and built-in contracts.")


def check_cli_help() -> None:
    commands = [
        ["praxis_eval.scripts.setup", "--help"],
        ["praxis_eval.scripts.setup", "robocasa", "--help"],
        ["praxis_eval.scripts.setup", "simpler", "--help"],
        ["praxis_eval.scripts.setup", "mshab", "--help"],
        ["praxis_eval.scripts.verify", "--help"],
        ["praxis_eval.scripts.verify", "libero", "--help"],
        ["praxis_eval.scripts.verify", "robocasa", "--help"],
        ["praxis_eval.scripts.verify", "robomimic", "--help"],
        ["praxis_eval.scripts.verify", "metaworld", "--help"],
        ["praxis_eval.scripts.verify", "simpler", "--help"],
        ["praxis_eval.scripts.verify", "mshab", "--help"],
    ]
    for command in commands:
        run_python_module(command)


def main() -> None:
    check_python_fences()
    check_public_api_smoke()
    check_cli_help()


if __name__ == "__main__":
    main()
