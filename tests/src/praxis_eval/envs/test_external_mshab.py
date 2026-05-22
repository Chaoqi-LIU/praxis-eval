from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from praxis_eval import ActionSpec
from praxis_eval.envs.eval_registry import EvalDriverContext
from praxis_eval.envs.mshab import eval as mshab_eval
from praxis_eval.envs.mshab.eval import (
    MshabEnvConfig,
    MshabTaskSpec,
    build_mshab_eval_command,
    mshab_recording_note_for_task,
    resolve_mshab_task_specs,
    run_mshab_eval,
    summarize_mshab_runs,
)
from praxis_eval.envs.subprocess_runtime import (
    SubprocessTaskResult as _TaskRunResult,
)
from praxis_eval.envs.subprocess_runtime import (
    run_subprocess_task_commands,
    single_env_recording_context,
    without_recording_context,
)
from praxis_eval.managed_paths import managed_asset_dir


def _make_context(tmp_path: Path) -> EvalDriverContext:
    return EvalDriverContext(
        cfg=None,  # type: ignore[arg-type]
        seed=7,
        eval_mode="remote_grpc",
        eval_output_dir=tmp_path / "out",
        eval_media_dir=tmp_path / "media",
        num_eval_per_task=32,
        num_parallel_env=8,
        eval_record_episodes_per_task=2,
        eval_debug_verbose=False,
        eval_step_timeout_sec=None,
        eval_policy_kwargs={"decode_keep_k": 4},
        eval_device="cpu",
        server_address="127.0.0.1:50051",
        policy=None,
        policy_preprocessor=None,
        policy_postprocessor=None,
        env_preprocessor=None,
        env_postprocessor=None,
        phase_heartbeat=lambda _label: None,
        progress_heartbeat=lambda _label: None,
        action_spec=ActionSpec(shape=(13,), dtype="float32"),
    )


def test_resolve_mshab_task_specs_set_table_subset():
    cfg = MshabEnvConfig(task="set_table", task_ids=[0, 3, 5])
    specs = resolve_mshab_task_specs({"task": "set_table", "task_ids": [0, 3, 5]}, cfg)
    assert [spec.alias for spec in specs] == [
        "set_table_pick",
        "set_table_open_kitchen_counter",
        "set_table_close_kitchen_counter",
    ]
    assert [spec.policy_task for spec in specs] == [
        "set table: pick object",
        "set table: open drawer",
        "set table: close drawer",
    ]


def test_resolve_mshab_task_specs_clean_pick_place_selector():
    cfg = MshabEnvConfig(task="pick,place")
    specs = resolve_mshab_task_specs({"task": "pick,place"}, cfg)

    assert [spec.alias for spec in specs] == [
        "set_table_pick",
        "set_table_place",
    ]
    assert [spec.policy_task for spec in specs] == [
        "set table: pick object",
        "set table: place object",
    ]


def test_build_mshab_eval_command_carries_python_and_kwargs(
    monkeypatch, tmp_path: Path
):
    asset_dir = tmp_path / "assets" / "mshab" / "maniskill_assets"
    monkeypatch.setattr(mshab_eval, "default_mshab_ms_asset_dir", lambda: asset_dir)
    python_bin = tmp_path / "mshab-praxis" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    cfg = MshabEnvConfig(
        task="set_table", python_bin=str(python_bin), debug_max_control_steps=8
    )
    task_spec = resolve_mshab_task_specs({"task": "set_table"}, cfg)[0]
    context = _make_context(tmp_path)

    command = build_mshab_eval_command(
        cfg_obj=cfg,
        task_spec=task_spec,
        context=context,
        server_address="127.0.0.1:50051",
        metrics_output_path=tmp_path / "metrics.json",
        record_dir=tmp_path / "media",
    )

    assert command[0] == "env"
    assert command[1] == f"MS_ASSET_DIR={asset_dir}"
    assert command[2] == str(python_bin.resolve())
    assert command[3:6] == ["-u", "-m", "mshab.praxis_eval"]
    assert "--praxis-policy-kwargs-json" in command
    assert "--task-alias" in command
    assert command[command.index("--policy-task") + 1] == "set table: pick object"
    assert command[command.index("--split") + 1] == "train"
    assert "--ms-asset-dir" in command
    assert command[command.index("--obs-mode") + 1] == "rgb"
    assert command[command.index("--frame-stack") + 1] == "1"
    assert command[command.index("--debug-max-control-steps") + 1] == "8"
    asset_index = command.index("--ms-asset-dir") + 1
    assert command[asset_index] == str(asset_dir)
    assert command[command.index("--metrics-output-path") + 1] == str(
        (tmp_path / "metrics.json").resolve()
    )
    assert command[command.index("--record-dir") + 1] == str(
        (tmp_path / "media").resolve()
    )


def test_mshab_runtime_root_resolves_from_configured_python(
    monkeypatch,
    tmp_path: Path,
):
    python_bin = tmp_path / "mshab-praxis" / "bin" / "python"
    runtime_root = tmp_path / "runtime" / "mshab"
    package_root = runtime_root / "mshab"
    seen: list[tuple[Path, str]] = []

    def fake_package_root_for_python(python: Path, import_name: str) -> Path:
        seen.append((python, import_name))
        return package_root

    monkeypatch.setattr(
        mshab_eval,
        "package_root_for_python",
        fake_package_root_for_python,
    )

    assert (
        mshab_eval.resolve_mshab_runtime_root(python_bin=str(python_bin))
        == runtime_root
    )
    assert mshab_eval.default_mshab_ms_asset_dir() == managed_asset_dir("mshab")
    assert seen == [(python_bin.resolve(), "mshab")]


def test_mshab_eval_local_inproc_starts_local_policy_server(
    tmp_path: Path,
    monkeypatch,
):
    from praxis_eval.envs import subprocess_runtime

    cfg = MshabEnvConfig(task="set_table")
    context = replace(
        _make_context(tmp_path),
        eval_mode="local_inproc",
        server_address=None,
        policy=object(),
    )

    started: list[object] = []

    @contextmanager
    def fake_local_policy_server(*, policy, action_spec, device):
        started.append((policy, action_spec, device))
        yield "127.0.0.1:54321"

    monkeypatch.setattr(
        subprocess_runtime,
        "local_policy_server",
        fake_local_policy_server,
    )
    monkeypatch.setattr(
        subprocess_runtime,
        "run_subprocess_task_commands",
        lambda **kwargs: {"server_address": kwargs["server_address"]},
    )

    result = run_mshab_eval({"task": "set_table"}, cfg, context)

    assert result == {"server_address": "127.0.0.1:54321"}
    assert started == [(context.policy, context.action_spec, "cpu")]


def test_build_mshab_eval_command_absolutizes_subprocess_paths(
    monkeypatch, tmp_path: Path
):
    python_bin = tmp_path / "mshab-praxis" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    cfg = MshabEnvConfig(task="set_table", python_bin=str(python_bin))
    task_spec = resolve_mshab_task_specs({"task": "set_table"}, cfg)[0]

    command = build_mshab_eval_command(
        cfg_obj=cfg,
        task_spec=task_spec,
        context=_make_context(tmp_path),
        server_address="127.0.0.1:50051",
        metrics_output_path=Path("relative/metrics.json"),
        record_dir=Path("relative/media"),
    )

    assert Path(command[command.index("--metrics-output-path") + 1]).is_absolute()
    assert Path(command[command.index("--record-dir") + 1]).is_absolute()


def test_mshab_summary_aggregates_weighted_metrics(tmp_path: Path):
    runs = [
        _TaskRunResult(
            spec=MshabTaskSpec(
                alias="set_table_pick",
                subtask="pick",
                target="all",
                task_id=0,
                env_id="PickSubtaskTrain-v0",
                policy_task="Pick",
                task_description="pick",
            ),
            metrics={
                "success_once_rate": 1.0,
                "success_at_end_rate": 0.5,
                "avg_episode_length": 10.0,
                "avg_sum_reward": 2.0,
                "avg_return_per_step": 0.2,
                "n_episodes": 2,
                "eval_s": 4.0,
            },
            metrics_path=tmp_path / "pick.json",
            log_path=tmp_path / "pick.log",
            video_paths=["a.mp4"],
        ),
        _TaskRunResult(
            spec=MshabTaskSpec(
                alias="set_table_open_fridge",
                subtask="open",
                target="fridge",
                task_id=2,
                env_id="OpenSubtaskTrain-v0",
                policy_task="OpenFr",
                task_description="open fridge",
            ),
            metrics={
                "success_once_rate": 0.0,
                "success_at_end_rate": 0.0,
                "avg_episode_length": 30.0,
                "avg_sum_reward": 4.0,
                "avg_return_per_step": 0.1,
                "n_episodes": 1,
                "eval_s": 6.0,
            },
            metrics_path=tmp_path / "open.json",
            log_path=tmp_path / "open.log",
            video_paths=["b.mp4"],
        ),
    ]

    summary = summarize_mshab_runs(runs)
    assert summary["overall"]["n_episodes"] == 3.0
    assert summary["overall"]["success_rate"] == 2.0 / 3.0
    assert summary["overall"]["success_at_end_rate"] == 1.0 / 3.0
    assert summary["overall"]["avg_episode_length"] == (10.0 * 2 + 30.0) / 3.0
    assert summary["overall"]["avg_sum_reward"] == (2.0 * 2 + 4.0) / 3.0
    assert summary["overall"]["avg_return_per_step"] == (0.2 * 2 + 0.1) / 3.0
    assert summary["per_task"]["set_table/0"]["task_alias"] == "set_table_pick"
    assert summary["per_group"]["set_table"]["n_episodes"] == 3.0


def test_run_mshab_eval_warns_when_parallelism_exceeds_task_episode_count(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from praxis_eval.envs import subprocess_runtime

    cfg = MshabEnvConfig(task="set_table", python_bin="/tmp/mshab-praxis/bin/python")
    context = _make_context(tmp_path)
    context = replace(context, num_eval_per_task=8, num_parallel_env=64)

    monkeypatch.setattr(
        subprocess_runtime,
        "run_subprocess_task_commands",
        lambda **_kwargs: {"overall": {}, "per_group": {}, "per_task": {}},
    )

    result = run_mshab_eval({"task": "set_table"}, cfg, context)
    stdout = capsys.readouterr().out
    assert "MS-HAB note" in stdout
    assert "tasks run sequentially" in stdout
    assert result == {"overall": {}, "per_group": {}, "per_task": {}}


def test_recording_context_for_task_eval_uses_single_env_video_rerun(tmp_path: Path):
    context = _make_context(tmp_path)

    video_context = single_env_recording_context(context)

    assert video_context is not None
    assert video_context.num_parallel_env == 1
    assert video_context.num_eval_per_task == context.eval_record_episodes_per_task
    assert (
        video_context.eval_record_episodes_per_task
        == context.eval_record_episodes_per_task
    )


def test_mshab_subprocess_tasks_use_separate_single_env_video_pass(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from praxis_eval.envs import subprocess_runtime

    cfg = MshabEnvConfig(task="set_table", python_bin="/tmp/mshab-praxis/bin/python")
    progress_events: list[str] = []
    context = replace(
        _make_context(tmp_path), progress_heartbeat=progress_events.append
    )
    task_spec = resolve_mshab_task_specs({"task": "set_table"}, cfg)[0]
    build_calls: list[dict[str, Any]] = []

    def fake_build_mshab_eval_command(
        *,
        cfg_obj,
        task_spec,
        context,
        server_address,
        metrics_output_path,
        record_dir,
    ):
        build_calls.append(
            {
                "cfg_obj": cfg_obj,
                "task_spec": task_spec,
                "context": context,
                "server_address": server_address,
                "metrics_output_path": metrics_output_path,
                "record_dir": record_dir,
            }
        )
        return ["fake-eval-command"]

    def fake_run_logged_subprocess(*, command, cwd, stdout_path):
        del command, cwd
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok\n", encoding="utf-8")
        call = build_calls[len(run_calls)]
        call["metrics_output_path"].parent.mkdir(parents=True, exist_ok=True)
        call["metrics_output_path"].write_text(
            json.dumps(
                {
                    "success_once_rate": 0.5,
                    "success_at_end_rate": 0.25,
                    "avg_episode_length": 123.0,
                    "avg_sum_reward": 1.0,
                    "avg_return_per_step": 0.01,
                    "n_episodes": float(call["context"].num_eval_per_task),
                    "eval_s": 1.0,
                }
            ),
            encoding="utf-8",
        )
        if call["metrics_output_path"].name == "video_metrics.json":
            video_path = call["record_dir"] / "eval_episode_0.mp4"
            video_path.write_bytes(b"")
        run_calls.append(call)

    run_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        subprocess_runtime, "run_logged_subprocess", fake_run_logged_subprocess
    )

    result = run_subprocess_task_commands(
        heartbeat_prefix="mshab",
        output_subdir="mshab",
        task_specs=[task_spec],
        cfg_obj=cfg,
        context=context,
        server_address="127.0.0.1:50051",
        build_command=fake_build_mshab_eval_command,
        cwd=tmp_path,
        summarize_runs=summarize_mshab_runs,
        metrics_context_for_task=without_recording_context,
        recording_context_for_task=single_env_recording_context,
        recording_note_for_task=mshab_recording_note_for_task,
    )

    assert len(build_calls) == 2
    assert build_calls[0]["metrics_output_path"].name == "metrics.json"
    assert build_calls[0]["context"].num_parallel_env == context.num_parallel_env
    assert build_calls[0]["context"].eval_record_episodes_per_task == 0
    assert build_calls[1]["metrics_output_path"].name == "video_metrics.json"
    assert build_calls[1]["context"].num_parallel_env == 1
    assert (
        build_calls[1]["context"].num_eval_per_task
        == context.eval_record_episodes_per_task
    )
    assert result["per_task"]["set_table/0"]["video_paths"] == [
        str(context.eval_media_dir / task_spec.alias / "eval_episode_0.mp4")
    ]
    stderr = capsys.readouterr().err
    assert "Eval:" in stderr
    assert "32/32" in stderr
    assert "succ_rate=50.0%" in stderr
    assert "pbar_tick done=32/32" in progress_events
