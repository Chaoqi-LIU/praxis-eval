# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared subprocess-eval runtime helpers for external env families."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from praxis_eval.contracts import ActionSpec
    from praxis_eval.types import Observation, Policy

CwdResolver = Path | Callable[[], Path]
_TaskSpec = TypeVar("_TaskSpec")


def parse_host_port(server_address: str) -> tuple[str, int]:
    host, port = str(server_address).rsplit(":", 1)
    return host, int(port)


def resolve_required_python_bin(
    *,
    python_bin: str | None,
    display_name: str,
    runtime_name: str,
    setup_command: str,
) -> Path:
    """Resolve the external runtime Python path required by subprocess eval."""
    if python_bin:
        return Path(str(python_bin)).expanduser().resolve()

    raise ValueError(
        f"{display_name} evaluation requires env.python_bin to point at the "
        f"{runtime_name}. Run `{setup_command}` first, then pass "
        f"`env.python_bin=/path/to/{runtime_name.rsplit(' ', 1)[0]}/bin/python`."
    )


def package_root_for_python(python_bin: str | Path, import_name: str) -> Path:
    """Resolve an import package root inside a target Python runtime."""
    code = f"""
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec({import_name!r})
assert spec is not None
if spec.origin is not None:
    print(Path(spec.origin).resolve().parent)
else:
    locations = list(spec.submodule_search_locations or ())
    if not locations:
        raise RuntimeError("Could not resolve package root for {import_name}.")
    print(Path(locations[0]).resolve())
"""
    result = subprocess.run(
        [str(python_bin), "-c", code],
        text=True,
        capture_output=True,
        check=True,
    )
    return Path(result.stdout.strip()).expanduser().resolve()


def append_policy_kwargs_json(
    command: list[str],
    policy_kwargs: dict[str, Any],
) -> None:
    """Append policy kwargs in the JSON format external eval scripts expect."""
    if not policy_kwargs:
        return

    import json

    command.extend(
        [
            "--praxis-policy-kwargs-json",
            json.dumps(policy_kwargs, sort_keys=True),
        ]
    )


def append_video_recording_args(command: list[str], *, max_videos: int) -> None:
    """Append the common external-eval video recording flags."""
    if max_videos <= 0:
        command.append("--no-save-video")
    else:
        command.extend(["--max-videos", str(int(max_videos))])


def without_recording_context(context: Any) -> Any:
    """Return the high-parallelism metrics-pass context with recording disabled."""
    if int(getattr(context, "eval_record_episodes_per_task", 0)) <= 0:
        return context
    return replace(context, eval_record_episodes_per_task=0)


def single_env_recording_context(context: Any) -> Any | None:
    """Return a dedicated single-env video pass context when recording is enabled."""
    num_recorded = int(getattr(context, "eval_record_episodes_per_task", 0))
    if num_recorded <= 0:
        return None
    return replace(
        context,
        num_eval_per_task=num_recorded,
        num_parallel_env=1,
    )


@contextmanager
def local_policy_server(
    policy: Policy,
    *,
    action_spec: ActionSpec | None = None,
    device: str | Any,
    host: str = "127.0.0.1",
    startup_timeout_sec: float = 30.0,
) -> Iterator[str]:
    """Start a localhost policy server around an in-process eval policy."""
    from praxis_remote import PolicyClient, PolicyServer

    _ = device
    port = _free_port()
    server = PolicyServer(
        _EvalPolicyHandler(policy, action_spec=action_spec),
        host=host,
        port=port,
    )
    thread = threading.Thread(
        target=server.serve,
        kwargs={"block": True},
        name="praxis-eval-subprocess-policy-server",
        daemon=True,
    )
    thread.start()

    address = f"{host}:{port}"
    deadline = time.monotonic() + float(startup_timeout_sec)
    client: PolicyClient | None = None
    try:
        client = PolicyClient(host=host, port=port)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                ready, _info = client.health_check()
            except Exception as exc:  # pragma: no cover - transient startup race
                last_error = exc
                time.sleep(0.2)
                continue
            if ready:
                yield address
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"Timed out waiting for local eval policy server at {address}."
        ) from last_error
    finally:
        if client is not None:
            client.close()
        server.stop(grace=0)
        thread.join(timeout=5.0)


class _EvalPolicyHandler:
    """Adapt a ``praxis_eval.Policy`` to the praxis-remote server handler."""

    def __init__(self, policy: Policy, *, action_spec: ActionSpec | None) -> None:
        self.policy = policy
        self.action_spec = action_spec

    def predict_action(
        self,
        observations: Sequence[Observation],
        *,
        policy_kwargs: dict[str, Any] | None = None,
        episode_ids: Sequence[str] | None = None,
    ):
        return self.policy.act(
            observations,
            action_spec=self.action_spec,
            policy_kwargs=policy_kwargs,
            episode_ids=episode_ids,
        )

    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        self.policy.reset(episode_ids=episode_ids)

    def model_info(self) -> str:
        return self.policy.__class__.__name__


@dataclass
class SubprocessTaskResult:
    """One completed external-env task subprocess result."""

    spec: Any
    metrics: dict[str, Any]
    metrics_path: Path
    log_path: Path
    video_paths: list[str]


def raw_task_selectors(
    raw_cfg: dict[str, Any],
    cfg_obj: Any,
    *,
    default: str,
    label: str,
) -> list[str]:
    """Resolve comma-delimited or list-style task selectors from env config."""
    tasks_value = getattr(cfg_obj, "tasks", raw_cfg.get("tasks"))
    if tasks_value is not None:
        if not isinstance(tasks_value, (list, tuple)):
            raise TypeError(
                f"env.tasks must be a list/tuple of {label} task selectors "
                "when provided."
            )
        selectors = [str(item).strip() for item in tasks_value if str(item).strip()]
    else:
        task_value = str(getattr(cfg_obj, "task", raw_cfg.get("task", default)))
        selectors = [item.strip() for item in task_value.split(",") if item.strip()]
    if not selectors:
        raise ValueError(f"{label} env config must resolve at least one task selector.")
    return selectors


def apply_task_id_filter(
    task_specs: list[_TaskSpec],
    *,
    task_ids: Any,
    label: str,
) -> list[_TaskSpec]:
    """Apply optional task-id subsetting to a resolved ordered task list."""
    normalized = normalize_task_ids(task_ids)
    if normalized is None:
        return task_specs
    if not task_specs:
        raise ValueError(f"No {label} tasks resolved before applying task_ids.")
    for task_id in normalized:
        if task_id < 0 or task_id >= len(task_specs):
            raise ValueError(
                f"{label} task_id {task_id} out of range [0, {len(task_specs) - 1}]."
            )
    return [task_specs[index] for index in normalized]


def normalize_task_ids(task_ids: Any) -> list[int] | None:
    """Normalize optional task ids to sorted unique integer indexes."""
    if task_ids is None:
        return None
    if not isinstance(task_ids, (list, tuple, set)):
        raise TypeError("env.task_ids must be a list/tuple/set of integers.")
    return sorted({int(task_id) for task_id in task_ids})


def sum_metric(runs: list[SubprocessTaskResult], key: str) -> float:
    """Sum one numeric metric across subprocess task results."""
    return sum(float(run.metrics[key]) for run in runs)


def weighted_metric(runs: list[SubprocessTaskResult], key: str) -> float:
    """Average one numeric metric weighted by each run's episode count."""
    episodes = sum_metric(runs, "n_episodes")
    if episodes <= 0:
        return 0.0
    return (
        sum(float(run.metrics[key]) * float(run.metrics["n_episodes"]) for run in runs)
        / episodes
    )


@dataclass
class _EvalProgressTotals:
    """Global progress state for sequential external-env task subprocesses."""

    expected_total: int
    completed: int = 0
    successes: float = 0.0
    episode_lengths: float = 0.0

    @property
    def success_rate_pct(self) -> float:
        if self.completed <= 0:
            return 0.0
        return 100.0 * self.successes / float(self.completed)

    @property
    def avg_episode_length(self) -> float:
        if self.completed <= 0:
            return 0.0
        return self.episode_lengths / float(self.completed)

    def update(self, metrics: dict[str, Any]) -> int:
        completed = _metric_episode_count(metrics)
        self.completed += completed
        success_rate = _metric_success_rate(metrics)
        if success_rate is not None:
            self.successes += success_rate * completed
        avg_length = _metric_avg_episode_length(metrics)
        if avg_length is not None:
            self.episode_lengths += avg_length * completed
        return completed


def _metric_episode_count(metrics: dict[str, Any]) -> int:
    return max(0, int(round(float(metrics.get("n_episodes", 0)))))


def _metric_success_rate(metrics: dict[str, Any]) -> float | None:
    """Return the canonical 0..1 success metric from a task metrics payload."""
    for key in ("success_rate", "success_once_rate"):
        if key in metrics:
            return float(metrics[key])
    return None


def _metric_avg_episode_length(metrics: dict[str, Any]) -> float | None:
    if "avg_episode_length" not in metrics:
        return None
    return float(metrics["avg_episode_length"])


def run_subprocess_eval_flow(
    *,
    display_name: str,
    heartbeat_prefix: str,
    output_subdir: str,
    task_specs: Sequence[Any],
    cfg_obj: Any,
    context: Any,
    build_command: Callable[..., list[str]],
    cwd: CwdResolver,
    summarize_runs: Callable[[list[SubprocessTaskResult]], dict[str, Any]],
    metrics_context_for_task: Callable[[Any], Any] | None = None,
    recording_context_for_task: Callable[[Any], Any | None] | None = None,
    recording_note_for_task: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    """Run sequential external-env eval subprocesses behind a policy server."""
    if context.num_parallel_env > context.num_eval_per_task:
        print(
            f"{display_name} note: tasks run sequentially, so useful per-task "
            f"parallelism is capped by eval.num_eval_per_task "
            f"({context.num_eval_per_task}). "
            f"Current eval.num_parallel_env={context.num_parallel_env} is larger "
            "than needed for one task wave."
        )
    print(
        f"Prepared {len(task_specs)} {display_name} tasks for eval "
        f"(num_eval_per_task={context.num_eval_per_task}, "
        f"num_parallel_env={context.num_parallel_env}, subprocess=True)"
    )

    context.phase_heartbeat(f"{heartbeat_prefix}_tasks_resolved")
    if context.eval_mode == "remote_grpc":
        if not context.server_address:
            raise ValueError(
                "eval.mode=remote_grpc requires `+server_address=host:port`."
            )
        return run_subprocess_task_commands(
            heartbeat_prefix=heartbeat_prefix,
            output_subdir=output_subdir,
            task_specs=task_specs,
            cfg_obj=cfg_obj,
            context=context,
            server_address=context.server_address,
            build_command=build_command,
            cwd=cwd,
            summarize_runs=summarize_runs,
            metrics_context_for_task=metrics_context_for_task,
            recording_context_for_task=recording_context_for_task,
            recording_note_for_task=recording_note_for_task,
        )

    if context.policy is None:
        raise RuntimeError(f"{display_name} local evaluation requires a loaded policy.")
    with local_policy_server(
        policy=context.policy,
        action_spec=getattr(context, "action_spec", None),
        device=context.eval_device,
    ) as address:
        return run_subprocess_task_commands(
            heartbeat_prefix=heartbeat_prefix,
            output_subdir=output_subdir,
            task_specs=task_specs,
            cfg_obj=cfg_obj,
            context=context,
            server_address=address,
            build_command=build_command,
            cwd=cwd,
            summarize_runs=summarize_runs,
            metrics_context_for_task=metrics_context_for_task,
            recording_context_for_task=recording_context_for_task,
            recording_note_for_task=recording_note_for_task,
        )


def run_subprocess_task_commands(
    *,
    heartbeat_prefix: str,
    output_subdir: str,
    task_specs: Sequence[Any],
    cfg_obj: Any,
    context: Any,
    server_address: str,
    build_command: Callable[..., list[str]],
    cwd: CwdResolver,
    summarize_runs: Callable[[list[SubprocessTaskResult]], dict[str, Any]],
    metrics_context_for_task: Callable[[Any], Any] | None = None,
    recording_context_for_task: Callable[[Any], Any | None] | None = None,
    recording_note_for_task: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    """Run the per-task external subprocess commands and collect artifacts."""
    import json

    from tqdm import tqdm

    task_root = context.eval_output_dir / output_subdir
    task_root.mkdir(parents=True, exist_ok=True)
    resolved_cwd = cwd() if callable(cwd) else cwd
    runs: list[SubprocessTaskResult] = []
    progress = _EvalProgressTotals(
        expected_total=max(1, len(task_specs) * int(context.num_eval_per_task)),
    )
    progress_cm = tqdm(
        total=progress.expected_total,
        desc="Eval",
        unit="ep",
        dynamic_ncols=True,
    )
    progress_start = time.time()
    with progress_cm as pbar:
        for index, task_spec in enumerate(task_specs, start=1):
            task_alias = str(task_spec.alias)
            context.phase_heartbeat(
                f"{heartbeat_prefix}_task_begin "
                f"alias={task_alias} idx={index}/{len(task_specs)}"
            )
            task_dir = task_root / task_alias
            metrics_path = task_dir / "metrics.json"
            log_path = task_dir / "stdout.log"
            record_dir = context.eval_media_dir / task_alias
            record_dir.mkdir(parents=True, exist_ok=True)

            metrics_context = (
                metrics_context_for_task(context)
                if metrics_context_for_task is not None
                else context
            )
            command = build_command(
                cfg_obj=cfg_obj,
                task_spec=task_spec,
                context=metrics_context,
                server_address=server_address,
                metrics_output_path=metrics_path,
                record_dir=record_dir,
            )
            run_logged_subprocess(
                command=command,
                cwd=resolved_cwd,
                stdout_path=log_path,
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            video_context = (
                recording_context_for_task(context)
                if recording_context_for_task is not None
                else None
            )
            if video_context is not None:
                clear_recorded_videos(record_dir)
                if recording_note_for_task is not None:
                    print(recording_note_for_task(task_spec))
                video_metrics_path = task_dir / "video_metrics.json"
                video_log_path = task_dir / "video_stdout.log"
                video_command = build_command(
                    cfg_obj=cfg_obj,
                    task_spec=task_spec,
                    context=video_context,
                    server_address=server_address,
                    metrics_output_path=video_metrics_path,
                    record_dir=record_dir,
                )
                run_logged_subprocess(
                    command=video_command,
                    cwd=resolved_cwd,
                    stdout_path=video_log_path,
                )

            completed = progress.update(metrics)
            elapsed = max(time.time() - progress_start, 1e-6)
            pbar.set_postfix(
                succ_rate=f"{progress.success_rate_pct:.1f}%",
                avg_len=f"{progress.avg_episode_length:.1f}",
                ep_s=f"{(progress.completed / elapsed):.2f}",
                refresh=False,
            )
            pbar.update(completed)

            runs.append(
                SubprocessTaskResult(
                    spec=task_spec,
                    metrics=metrics,
                    metrics_path=metrics_path,
                    log_path=log_path,
                    video_paths=discover_video_paths(record_dir),
                )
            )
            context.progress_heartbeat(
                f"pbar_tick done={progress.completed}/{progress.expected_total}"
            )
            context.progress_heartbeat(
                f"{heartbeat_prefix}_task_done "
                f"alias={task_alias} idx={index}/{len(task_specs)}"
            )
    return summarize_runs(runs)


def clear_recorded_videos(record_dir: Path) -> None:
    """Remove stale videos before a dedicated recording rerun."""
    for path in record_dir.rglob("*.mp4"):
        path.unlink()


def discover_video_paths(record_dir: Path) -> list[str]:
    """Return recorded mp4 paths for one external-env task."""
    if not record_dir.exists():
        return []
    return sorted(str(path) for path in record_dir.rglob("*.mp4"))


def run_logged_subprocess(
    *,
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    on_output_line: Callable[[str], None] | None = None,
) -> None:
    """Run a subprocess, streaming combined stdout/stderr to ``stdout_path``."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    tail_lines: deque[str] = deque(maxlen=40)
    with stdout_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            sys.stderr.write(line)
            sys.stderr.flush()
            stripped = line.rstrip("\n")
            if stripped:
                tail_lines.append(stripped)
                if on_output_line is not None:
                    on_output_line(stripped)
        proc.stdout.close()
        result = proc.wait()
    if result == 0:
        return
    tail = "\n".join(tail_lines)
    raise RuntimeError(
        f"Eval subprocess failed with exit code {result}.\n"
        f"Command: {' '.join(command)}\n"
        f"Log: {stdout_path}\n"
        f"Last output:\n{tail}"
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
