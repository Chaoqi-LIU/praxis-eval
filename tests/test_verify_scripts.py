from __future__ import annotations

import subprocess
import sys


def test_setup_and_verify_script_help_entry_points_import_without_env_extras() -> None:
    modules = [
        "praxis_eval.scripts.setup",
        "praxis_eval.scripts.setup_mshab",
        "praxis_eval.scripts.setup_robocasa",
        "praxis_eval.scripts.setup_simpler",
        "praxis_eval.scripts.verify",
        "praxis_eval.scripts.verify_libero",
        "praxis_eval.scripts.verify_simpler",
        "praxis_eval.scripts.verify_mshab",
        "praxis_eval.scripts.verify_metaworld",
        "praxis_eval.scripts.verify_robocasa",
        "praxis_eval.scripts.verify_robomimic",
    ]
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            text=True,
            capture_output=True,
            check=True,
        )
        assert "usage:" in result.stdout


def test_dispatch_scripts_forward_help_to_selected_benchmark() -> None:
    commands = [
        ("praxis_eval.scripts.setup", "robocasa"),
        ("praxis_eval.scripts.verify", "libero"),
    ]
    for module, benchmark in commands:
        result = subprocess.run(
            [sys.executable, "-m", module, benchmark, "--help"],
            text=True,
            capture_output=True,
            check=True,
        )
        assert "usage: praxis-eval-" in result.stdout
        assert benchmark in result.stdout
