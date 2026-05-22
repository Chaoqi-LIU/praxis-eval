"""Tests for eval phase watchdog diagnostics helpers."""

from __future__ import annotations

import time

from praxis_eval import EvalPhaseWatchdog


def test_phase_watchdog_heartbeats_do_not_crash() -> None:
    watchdog = EvalPhaseWatchdog(threshold_sec=30.0)
    watchdog.mark_phase("unit_test_phase")
    watchdog.mark_progress("unit_test_progress")


def test_phase_watchdog_start_stop_is_deterministic() -> None:
    watchdog = EvalPhaseWatchdog(threshold_sec=60.0)
    assert watchdog.is_running is False
    watchdog.start()
    assert watchdog.is_running is True
    time.sleep(0.05)
    watchdog.stop(join_timeout_sec=1.0)
    assert watchdog.is_running is False
