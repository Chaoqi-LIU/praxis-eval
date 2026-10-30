"""Tests for LIBERO stdout/stderr suppression."""

from __future__ import annotations

import sys
import threading

import pytest

from praxis_eval.envs.libero.output import suppress_libero_output


def test_overlapping_suppressions_restore_the_original_streams(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stdout, stderr = sys.stdout, sys.stderr
    # Restores capsys's streams, before capsys tears down, if suppression leaks.
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    first_entered = threading.Event()
    second_entered = threading.Event()
    first_exited = threading.Event()

    def first() -> None:
        with suppress_libero_output(True):
            first_entered.set()
            second_entered.wait()
        first_exited.set()

    def second() -> None:
        first_entered.wait()
        with suppress_libero_output(True):
            second_entered.set()
            first_exited.wait()
            print("still suppressed")

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sys.stdout is stdout and sys.stderr is stderr
    assert "still suppressed" not in capsys.readouterr().out
