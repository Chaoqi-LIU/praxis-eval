# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Dispatch benchmark verification helpers."""

from __future__ import annotations

from praxis_eval.scripts._dispatch import dispatch_command

VERIFY_MODULES = {
    "libero": "praxis_eval.scripts.verify_libero",
    "metaworld": "praxis_eval.scripts.verify_metaworld",
    "mshab": "praxis_eval.scripts.verify_mshab",
    "robocasa": "praxis_eval.scripts.verify_robocasa",
    "robocasa_gr1": "praxis_eval.scripts.verify_robocasa_gr1",
    "robomimic": "praxis_eval.scripts.verify_robomimic",
    "simpler": "praxis_eval.scripts.verify_simpler",
}


def main() -> None:
    dispatch_command(
        program="praxis-eval-verify",
        description="Run short benchmark rollouts to verify installed runtimes.",
        modules=VERIFY_MODULES,
    )


if __name__ == "__main__":
    main()
