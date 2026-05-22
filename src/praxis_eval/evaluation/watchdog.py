# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Eval-phase watchdog and heartbeat helpers."""

from __future__ import annotations

import datetime as dt
import faulthandler
import logging
import os
import threading
import time
from pathlib import Path

_LOGGER = logging.getLogger(__name__)
_DEFAULT_WATCHDOG_SEC = 180.0


def _wall_clock_iso(now_ts: float | None = None) -> str:
    timestamp = time.time() if now_ts is None else float(now_ts)
    return (
        dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds")
    )


def resolve_phase_watchdog_threshold_sec(raw: object) -> float | None:
    """Return watchdog threshold in seconds from config, or ``None`` when disabled."""
    if raw is None:
        return None

    value = str(raw).strip()
    if value == "":
        return None

    try:
        threshold_sec = float(value)
    except ValueError:
        _LOGGER.warning(
            "WATCHDOG invalid threshold=%r; defaulting to %.1fs.",
            raw,
            _DEFAULT_WATCHDOG_SEC,
        )
        threshold_sec = _DEFAULT_WATCHDOG_SEC

    if threshold_sec <= 0.0:
        return None
    return float(threshold_sec)


class EvalPhaseWatchdog:
    """Observe eval phase/progress liveness and dump best-effort diagnostics."""

    def __init__(
        self,
        *,
        threshold_sec: float,
        process_start_monotonic: float | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if threshold_sec <= 0.0:
            raise ValueError(f"threshold_sec must be > 0, got {threshold_sec}")
        self._threshold_sec = float(threshold_sec)
        self._logger = _LOGGER if logger is None else logger
        self._process_start_monotonic = (
            time.monotonic()
            if process_start_monotonic is None
            else float(process_start_monotonic)
        )
        self._poll_sec = min(5.0, max(1.0, self._threshold_sec / 6.0))

        now_mono = time.monotonic()
        self._phase_label = "eval.main.start"
        self._last_phase_monotonic = now_mono
        self._last_progress_monotonic = now_mono
        self._last_dump_monotonic = 0.0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def process_start_monotonic(self) -> float:
        return self._process_start_monotonic

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def mark_phase(self, label: str) -> None:
        now_mono = time.monotonic()
        with self._lock:
            self._phase_label = str(label)
            self._last_phase_monotonic = now_mono
        self._logger.info(
            "PHASE label=%s pid=%d wall=%s mono_elapsed=%.3fs",
            label,
            os.getpid(),
            _wall_clock_iso(),
            now_mono - self._process_start_monotonic,
        )

    def mark_progress(self, label: str) -> None:
        now_mono = time.monotonic()
        with self._lock:
            self._last_progress_monotonic = now_mono
        self._logger.info(
            "HEARTBEAT label=%s pid=%d wall=%s mono_elapsed=%.3fs",
            label,
            os.getpid(),
            _wall_clock_iso(),
            now_mono - self._process_start_monotonic,
        )

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._watch_loop,
                name="praxis-eval-phase-watchdog",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, join_timeout_sec: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=join_timeout_sec)

    def _watch_loop(self) -> None:
        while not self._stop_event.wait(self._poll_sec):
            now_mono = time.monotonic()
            with self._lock:
                phase_label = self._phase_label
                phase_idle_sec = now_mono - self._last_phase_monotonic
                progress_idle_sec = now_mono - self._last_progress_monotonic
                since_last_dump = now_mono - self._last_dump_monotonic

            if (
                phase_idle_sec < self._threshold_sec
                and progress_idle_sec < self._threshold_sec
            ):
                continue
            if since_last_dump < self._threshold_sec:
                continue

            with self._lock:
                self._last_dump_monotonic = now_mono
            self._emit_watchdog_dump(
                phase_label=phase_label,
                phase_idle_sec=phase_idle_sec,
                progress_idle_sec=progress_idle_sec,
            )

    def _emit_watchdog_dump(
        self,
        *,
        phase_label: str,
        phase_idle_sec: float,
        progress_idle_sec: float,
    ) -> None:
        self._logger.warning(
            "WATCHDOG fired pid=%d wall=%s phase=%s phase_idle=%.1fs progress_idle=%.1fs threshold=%.1fs",
            os.getpid(),
            _wall_clock_iso(),
            phase_label,
            phase_idle_sec,
            progress_idle_sec,
            self._threshold_sec,
        )

        self._emit_memory_snapshot()
        self._logger.warning("WATCHDOG main-process faulthandler dump follows.")
        faulthandler.dump_traceback(all_threads=True)

    def _emit_memory_snapshot(self) -> None:
        status_path = Path("/proc/self/status")
        meminfo_path = Path("/proc/meminfo")

        status_summary = "missing"
        try:
            lines = status_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            keep_keys = ("VmRSS", "VmHWM", "VmSize", "Threads")
            picked = [line.strip() for line in lines if line.startswith(keep_keys)]
            status_summary = ", ".join(picked) if picked else "no Vm* fields"
        except Exception as exc:
            status_summary = f"error={exc!r}"

        mem_available = "missing"
        try:
            for line in meminfo_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.startswith("MemAvailable:"):
                    mem_available = line.strip()
                    break
        except Exception as exc:
            mem_available = f"error={exc!r}"

        self._logger.warning(
            "WATCHDOG memory /proc/self/status=%s | /proc/meminfo=%s",
            status_summary,
            mem_available,
        )
