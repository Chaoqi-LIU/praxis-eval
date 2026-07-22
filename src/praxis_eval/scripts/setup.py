# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Dispatch benchmark setup helpers."""

from __future__ import annotations

from praxis_eval.scripts._dispatch import dispatch_command

SETUP_MODULES = {
    "mshab": "praxis_eval.scripts.setup_mshab",
    "robocasa": "praxis_eval.scripts.setup_robocasa",
    "robocasa_gr1": "praxis_eval.scripts.setup_robocasa_gr1",
    "simpler": "praxis_eval.scripts.setup_simpler",
}


def main() -> None:
    dispatch_command(
        program="praxis-eval-setup",
        description="Prepare benchmark assets or dedicated simulator runtimes.",
        modules=SETUP_MODULES,
    )


if __name__ == "__main__":
    main()
