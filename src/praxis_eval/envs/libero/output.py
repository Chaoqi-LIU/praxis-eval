# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO-owned stdout/stderr suppression helpers."""

from __future__ import annotations

import contextlib
import os


@contextlib.contextmanager
def suppress_libero_output(enabled: bool):
    """Best-effort suppression for noisy upstream LIBERO stdout/stderr prints."""
    if not enabled:
        yield
        return
    with (
        open(os.devnull, "w") as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
    ):
        yield
