from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from praxis_remote import PolicyClient
from praxis_remote.serialization import build_observation

from praxis_eval import ActionSpec
from praxis_eval.envs.eval_registry import EvalDriverContext
from praxis_eval.envs.simpler import eval as simpler_eval
from praxis_eval.envs.simpler.eval import (
    SimplerEnvConfig,
    SimplerTaskSpec,
    build_simpler_eval_command,
    resolve_simpler_task_specs,
    run_simpler_eval,
    summarize_simpler_runs,
)
from praxis_eval.envs.subprocess_runtime import (
    SubprocessTaskResult as _TaskRunResult,
)
from praxis_eval.envs.subprocess_runtime import (
    local_policy_server,
    single_env_recording_context,
    without_recording_context,
)
from praxis_eval.managed_paths import managed_asset_dir


class _DummyPolicy:
    def __init__(self) -> None:
        self.action_specs = []

    def reset(self, episode_ids=None) -> None:
        _ = episode_ids

    def act(
        self,
        observations,
        *,
        action_spec=None,
        policy_kwargs=None,
        episode_ids=None,
    ) -> np.ndarray:
        self.action_specs.append(action_spec)
        _ = (policy_kwargs, episode_ids)
        actions = []
        for observation in observations:
            state = np.asarray(observation["observation.state"], dtype=np.float32)
            actions.append(state[:7])
        return np.stack(actions, axis=0)


def _make_context(tmp_path: Path) -> EvalDriverContext:
    return EvalDriverContext(
        cfg=None,  # type: ignore[arg-type]
        seed=17,
        eval_mode="remote_grpc",
        eval_output_dir=tmp_path / "out",
        eval_media_dir=tmp_path / "media",
        num_eval_per_task=50,
        num_parallel_env=16,
        eval_record_episodes_per_task=3,
        eval_debug_verbose=False,
        eval_step_timeout_sec=None,
        eval_policy_kwargs={"decode_keep_k": 4, "nested": {"foo": "bar"}},
        eval_device="cpu",
        server_address="127.0.0.1:50051",
        policy=None,
        policy_preprocessor=None,
        policy_postprocessor=None,
        env_preprocessor=None,
        env_postprocessor=None,
        phase_heartbeat=lambda _label: None,
        progress_heartbeat=lambda _label: None,
    )


def test_resolve_simpler_task_specs_bridge_subset():
    cfg = SimplerEnvConfig(task="bridge", task_ids=[0, 2])
    specs = resolve_simpler_task_specs({"task": "bridge", "task_ids": [0, 2]}, cfg)
    assert [spec.alias for spec in specs] == [
        "widowx_carrot_on_plate",
        "widowx_stack_cube",
    ]


def test_build_simpler_eval_command_carries_python_and_kwargs(
    monkeypatch, tmp_path: Path
):
    asset_dir = tmp_path / "assets" / "simpler" / "maniskill_assets"
    monkeypatch.setattr(simpler_eval, "default_simpler_ms_asset_dir", lambda: asset_dir)
    python_bin = tmp_path / "simpler-praxis" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    cfg = SimplerEnvConfig(
        task="bridge",
        python_bin=str(python_bin),
    )
    task_spec = resolve_simpler_task_specs({"task": "bridge"}, cfg)[0]
    context = _make_context(tmp_path)

    command = build_simpler_eval_command(
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
    assert command[3:6] == ["-u", "-m", "simpler_env.real2sim_eval_maniskill3"]
    assert "--praxis-policy-kwargs-json" in command
    assert "--metrics-output-path" in command
    assert "--praxis-primary-image-key" in command
    assert command[command.index("--praxis-primary-image-key") + 1] == (
        "observation.images.image"
    )
    assert "--praxis-state-key" in command
    assert command[command.index("--praxis-state-key") + 1] == "observation.state"
    assert Path(command[command.index("--metrics-output-path") + 1]).is_absolute()
    assert Path(command[command.index("--record-dir") + 1]).is_absolute()
    assert "--max-videos" in command
    assert "--env-id" in command
    assert task_spec.env_id in command


def test_simpler_runtime_root_resolves_from_configured_python(
    monkeypatch,
    tmp_path: Path,
):
    python_bin = tmp_path / "simpler-praxis" / "bin" / "python"
    runtime_root = tmp_path / "runtime" / "SimplerEnv"
    package_root = runtime_root / "simpler_env"
    seen: list[tuple[Path, str]] = []

    def fake_package_root_for_python(python: Path, import_name: str) -> Path:
        seen.append((python, import_name))
        return package_root

    monkeypatch.setattr(
        simpler_eval,
        "package_root_for_python",
        fake_package_root_for_python,
    )

    assert (
        simpler_eval.resolve_simpler_runtime_root(python_bin=str(python_bin))
        == runtime_root
    )
    assert simpler_eval.default_simpler_ms_asset_dir() == managed_asset_dir("simpler")
    assert seen == [(python_bin.resolve(), "simpler_env")]


def test_local_policy_server_serves_predictions():
    policy = _DummyPolicy()
    action_spec = ActionSpec(shape=(7,), dtype="float32")
    with local_policy_server(policy, action_spec=action_spec, device="cpu") as address:
        host, port = address.rsplit(":", 1)
        client = PolicyClient(host=host, port=int(port))
        try:
            ready, _info = client.health_check()
            assert ready is True
            action = client.predict_action(
                [
                    build_observation(
                        state=np.arange(8, dtype=np.float32),
                        task_description="stack the block",
                    )
                ]
            )
        finally:
            client.close()
    np.testing.assert_allclose(action, np.arange(7, dtype=np.float32)[None, :])
    assert policy.action_specs == [action_spec]


def test_simpler_recording_context_uses_single_env_rerun(tmp_path: Path):
    context = _make_context(tmp_path)

    metrics_context = without_recording_context(context)
    recording_context = single_env_recording_context(context)

    assert metrics_context.eval_record_episodes_per_task == 0
    assert recording_context is not None
    assert recording_context.num_eval_per_task == context.eval_record_episodes_per_task
    assert recording_context.num_parallel_env == 1


def test_simpler_recording_context_disabled_when_no_videos(tmp_path: Path):
    context = replace(_make_context(tmp_path), eval_record_episodes_per_task=0)

    assert without_recording_context(context) is context
    assert single_env_recording_context(context) is None


def test_simpler_summary_aggregates_weighted_metrics(tmp_path: Path):
    runs = [
        _TaskRunResult(
            spec=SimplerTaskSpec(
                alias="widowx_carrot_on_plate",
                env_id="PutCarrotOnPlateInScene-v1",
                task_id=0,
            ),
            metrics={
                "success_rate": 1.0,
                "avg_episode_length": 10.0,
                "avg_reward": 2.0,
                "avg_max_reward": 1.0,
                "n_episodes": 2,
                "eval_s": 4.0,
                "task_description": "put carrot on plate",
                "task_descriptions": ["put carrot on plate"],
            },
            metrics_path=tmp_path / "carrot.json",
            log_path=tmp_path / "carrot.log",
            video_paths=["a.mp4"],
        ),
        _TaskRunResult(
            spec=SimplerTaskSpec(
                alias="widowx_spoon_on_towel",
                env_id="PutSpoonOnTableClothInScene-v1",
                task_id=1,
            ),
            metrics={
                "success_rate": 0.0,
                "avg_episode_length": 30.0,
                "avg_reward": 4.0,
                "avg_max_reward": 3.0,
                "n_episodes": 1,
                "eval_s": 6.0,
            },
            metrics_path=tmp_path / "spoon.json",
            log_path=tmp_path / "spoon.log",
            video_paths=["b.mp4"],
        ),
    ]

    summary = summarize_simpler_runs(runs)
    assert summary["overall"]["n_episodes"] == 3.0
    assert summary["overall"]["success_rate"] == 2.0 / 3.0
    assert summary["overall"]["avg_episode_length"] == (10.0 * 2 + 30.0) / 3.0
    assert summary["overall"]["avg_reward"] == (2.0 * 2 + 4.0) / 3.0
    assert summary["overall"]["avg_max_reward"] == (1.0 * 2 + 3.0) / 3.0
    assert summary["per_task"]["bridge/0"]["task_alias"] == "widowx_carrot_on_plate"
    assert summary["per_task"]["bridge/0"]["task_description"] == "put carrot on plate"
    assert summary["per_task"]["bridge/0"]["task_descriptions"] == [
        "put carrot on plate"
    ]
    assert summary["per_group"]["bridge"]["n_episodes"] == 3.0


def test_run_simpler_eval_warns_when_parallelism_exceeds_task_episode_count(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from praxis_eval.envs import subprocess_runtime

    cfg = SimplerEnvConfig(task="bridge", python_bin="/tmp/simpler-praxis/bin/python")
    context = _make_context(tmp_path)
    context = replace(context, num_eval_per_task=50, num_parallel_env=64)

    monkeypatch.setattr(
        subprocess_runtime,
        "run_subprocess_task_commands",
        lambda **_kwargs: {"overall": {}, "per_group": {}, "per_task": {}},
    )

    result = run_simpler_eval({"task": "bridge"}, cfg, context)
    stdout = capsys.readouterr().out
    assert "SIMPLER note" in stdout
    assert "tasks run sequentially" in stdout
    assert result == {"overall": {}, "per_group": {}, "per_task": {}}
