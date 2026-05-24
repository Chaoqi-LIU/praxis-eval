from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from praxis_eval import EvalConfig, LocalPolicy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / ".tmp/praxis_eval_examples"


def test_local_policy_example_runs() -> None:
    _clean_output("01_minimal_local_policy")
    _run_script("examples/01_minimal_local_policy/run.py")
    _assert_results("01_minimal_local_policy", expected_episodes=3)


def test_custom_policy_adapter_example_runs() -> None:
    _clean_output("03_custom_policy_adapter")
    _run_script("examples/03_custom_policy_adapter/run.py")
    _assert_results(
        "03_custom_policy_adapter",
        expected_episodes=2,
        expected_success_rate=1.0,
    )


def test_custom_benchmark_driver_example_runs() -> None:
    _clean_output("04_custom_benchmark_driver")
    _run_script("examples/04_custom_benchmark_driver/run.py")
    _assert_results(
        "04_custom_benchmark_driver",
        expected_episodes=1,
        expected_success_rate=1.0,
    )


def test_custom_benchmark_driver_respects_num_eval_per_task(tmp_path: Path) -> None:
    driver = _load_line_reach_driver()
    result = driver.evaluate(
        policy=LocalPolicy(_line_reach_policy),
        config=EvalConfig(
            task="move right on the line",
            num_eval_per_task=3,
            output_dir=tmp_path,
            env_kwargs={"start": -1.0, "target": 1.0},
        ),
    )

    assert result.overall["n_episodes"] == 3
    assert result.overall["success_rate"] == 1.0
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert len(payload["episodes"]) == 3


def test_remote_policy_example_runs() -> None:
    _clean_output("02_remote_policy")
    port = _free_port()
    server = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "examples/02_remote_policy/serve_policy.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port(port)
        _run_script(
            "examples/02_remote_policy/run_eval.py",
            "--address",
            f"127.0.0.1:{port}",
        )
        _assert_results(
            "02_remote_policy",
            expected_episodes=2,
            expected_success_rate=1.0,
        )
    finally:
        server.terminate()
        try:
            server.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.communicate(timeout=5)


def _line_reach_policy(observations, *, policy_kwargs=None, episode_ids=None):
    del policy_kwargs, episode_ids
    actions = []
    for observation in observations:
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        target = np.asarray(observation["observation.goal"], dtype=np.float32)
        actions.append(np.clip(target - state, -1.0, 1.0))
    return np.stack(actions).astype(np.float32)


def _load_line_reach_driver():
    path = ROOT / "examples/04_custom_benchmark_driver/driver.py"
    spec = importlib.util.spec_from_file_location("line_reach_example_driver", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LineReachDriver()


def _run_script(path: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / path), *args],
        cwd=ROOT,
        env=_env(),
        check=True,
        timeout=30,
    )


def _clean_output(name: str) -> None:
    shutil.rmtree(OUTPUT_ROOT / name, ignore_errors=True)


def _assert_results(
    name: str,
    *,
    expected_episodes: int,
    expected_success_rate: float | None = None,
) -> None:
    path = OUTPUT_ROOT / name / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    overall = payload["overall"]
    assert overall["n_episodes"] == expected_episodes
    assert len(payload["episodes"]) == expected_episodes
    assert "avg_episode_length" in overall
    if expected_success_rate is not None:
        assert overall["success_rate"] == expected_success_rate


def _env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"server did not open port {port}")
