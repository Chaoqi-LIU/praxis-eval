# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO-owned stdout/stderr suppression helpers."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Any

# sys.stdout and sys.stderr are process-wide, so overlapping suppressions (nested,
# or from concurrent evaluate() calls in threads) share one redirect: the first
# entry installs it and the last exit restores the original streams.
_LOCK = threading.Lock()
_users = 0
_saved: tuple[Any, Any, Any] | None = None


@contextlib.contextmanager
def suppress_libero_output(enabled: bool):
    """Best-effort suppression for noisy upstream LIBERO stdout/stderr prints."""
    if not enabled:
        yield
        return
    _acquire()
    try:
        yield
    finally:
        _release()


def _acquire() -> None:
    global _users, _saved
    with _LOCK:
        if _users == 0:
            devnull = open(os.devnull, "w")  # noqa: SIM115
            _saved = (sys.stdout, sys.stderr, devnull)
            sys.stdout = sys.stderr = devnull
        _users += 1


def _release() -> None:
    global _users, _saved
    with _LOCK:
        _users -= 1
        if _users == 0:
            assert _saved is not None
            sys.stdout, sys.stderr, devnull = _saved
            _saved = None
            devnull.close()
