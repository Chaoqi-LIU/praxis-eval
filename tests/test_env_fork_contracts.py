from __future__ import annotations

import ast
import importlib
import importlib.metadata
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _load_python_file(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _distribution_file(distribution_name: str, relative_path: str) -> Path:
    try:
        dist = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"{distribution_name} is not installed.")
    path = Path(dist.locate_file(relative_path))
    if not path.exists():
        pytest.fail(f"{distribution_name} does not ship {relative_path}.")
    return path


def _load_simpler_remote_model_module() -> ModuleType:
    return _load_python_file(
        "simpler_praxis_remote_model_test",
        _distribution_file(
            "praxis-simpler",
            "simpler_env/policies/praxis/remote_model.py",
        ),
    )


def _load_simpler_bridge_state_module() -> ModuleType:
    return _load_python_file(
        "simpler_praxis_bridge_state_test",
        _distribution_file(
            "praxis-simpler",
            "simpler_env/policies/praxis/bridge_state.py",
        ),
    )


def _load_mshab_praxis_eval_module():
    pytest.importorskip("torch")
    pytest.importorskip("praxis_remote")
    return _load_python_file(
        "mshab_praxis_eval_test",
        _distribution_file("praxis-mshab", "mshab/praxis_eval.py"),
    )


def test_simpler_metric_flattening_handles_ragged_batches() -> None:
    eval_module_path = _distribution_file(
        "praxis-simpler", "simpler_env/real2sim_eval_maniskill3.py"
    )
    source_tree = ast.parse(eval_module_path.read_text(encoding="utf-8"))
    helper_nodes = [
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_flatten_metric_values", "_mean_metric"}
    ]
    namespace = {"np": np}
    exec(  # noqa: S102 - execute two parsed local helper definitions for coverage.
        compile(
            ast.Module(body=helper_nodes, type_ignores=[]),
            str(eval_module_path),
            "exec",
        ),
        namespace,
    )

    values = [
        np.array([1.0] * 16, dtype=np.float32),
        np.array([0.0] * 16, dtype=np.float32),
        np.array([1.0] * 16, dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]

    flattened = namespace["_flatten_metric_values"](values)

    assert flattened.shape == (50,)
    assert namespace["_mean_metric"]({"success": values}, "success") == pytest.approx(
        33.0 / 50.0
    )


def test_simpler_praxis_wrapper_preserves_uint8_image_transport() -> None:
    remote_model = _load_simpler_remote_model_module()
    image = np.arange(2 * 4 * 5 * 4, dtype=np.uint8).reshape(2, 4, 5, 4)

    prepared = remote_model._prepare_image_array(image)

    assert prepared.dtype == np.uint8
    assert prepared.shape == (2, 3, 4, 5)
    np.testing.assert_array_equal(prepared, np.moveaxis(image[..., :3], -1, 1))


def test_simpler_praxis_wrapper_float_images_remain_float32() -> None:
    remote_model = _load_simpler_remote_model_module()
    image = np.full((4, 5, 3), 0.5, dtype=np.float64)

    prepared = remote_model._prepare_image_array(image)

    assert prepared.dtype == np.float32
    assert prepared.shape == (3, 4, 5)
    np.testing.assert_allclose(prepared, np.moveaxis(image, -1, 0))


def test_simpler_praxis_wrapper_builds_bridge_observation_keys(monkeypatch) -> None:
    remote_model = _load_simpler_remote_model_module()

    class _FakePolicyClient:
        def __init__(self, *, host: str, port: int) -> None:
            self.host = host
            self.port = port

    monkeypatch.setitem(
        sys.modules,
        "praxis_remote",
        SimpleNamespace(PolicyClient=_FakePolicyClient),
    )
    wrapper = remote_model.PraxisRemoteInference(
        primary_image_key="observation.images.image",
        state_key="observation.state",
    )

    observations, batched_input, _device = wrapper._build_observations(
        image=np.zeros((4, 5, 3), dtype=np.uint8),
        state=np.arange(7, dtype=np.float32),
        images=None,
        task_description="put the carrot on the plate",
    )

    assert batched_input is False
    assert len(observations) == 1
    assert set(observations[0]) == {
        "observation.images.image",
        "observation.state",
        "task",
    }
    assert observations[0]["observation.images.image"].shape == (3, 4, 5)
    np.testing.assert_allclose(
        observations[0]["observation.state"], np.arange(7, dtype=np.float32)
    )


def test_simpler_praxis_wrapper_maps_bridge_action_to_sim_range() -> None:
    remote_model = _load_simpler_remote_model_module()
    wrapper = object.__new__(remote_model.PraxisRemoteInference)
    wrapper.policy_setup = "widowx_bridge"
    wrapper.action_scale = 1.0

    action = np.array(
        [
            [0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 0.0],
            [0.01, -0.02, 0.03, 0.2, 0.3, -0.4, 0.25],
            [0.01, -0.02, 0.03, 0.2, 0.3, -0.4, 0.75],
            [0.01, -0.02, 0.03, 0.2, 0.3, -0.4, 1.0],
        ],
        dtype=np.float32,
    )

    _raw_action, env_action = wrapper._format_action(action, return_device=None)

    np.testing.assert_allclose(env_action["world_vector"], action[:, :3])
    np.testing.assert_allclose(
        env_action["rotation_delta"],
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.3, -0.4],
            [0.2, 0.3, -0.4],
            [0.2, 0.3, -0.4],
        ],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        env_action["gripper"],
        [[-1.0], [-1.0], [1.0], [1.0]],
    )


def test_simpler_praxis_wrapper_rejects_malformed_bridge_actions() -> None:
    remote_model = _load_simpler_remote_model_module()
    wrapper = object.__new__(remote_model.PraxisRemoteInference)
    wrapper.policy_setup = "widowx_bridge"
    wrapper.action_scale = 1.0

    with pytest.raises(ValueError, match="Expected Bridge action shape"):
        wrapper._format_action(np.zeros((1, 6), dtype=np.float32), return_device=None)


def test_simpler_bridge_gripper_state_uses_normalized_open_fraction() -> None:
    torch = pytest.importorskip("torch")
    bridge_state = _load_simpler_bridge_state_module()
    qpos = torch.tensor(
        [
            [0.0, 0.0, 0.037, 0.037],
            [0.0, 0.0, 0.0185, 0.0185],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    qlimits = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.037],
            [0.0, 0.037],
        ],
        dtype=torch.float32,
    )

    gripper = bridge_state.normalize_bridge_gripper_qpos(qpos, qlimits)

    np.testing.assert_allclose(
        gripper.detach().cpu().numpy(),
        [[1.0], [0.5], [0.0]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_mshab_runtime_bootstrap_sets_asset_env_before_import(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bootstrap_path = _distribution_file("praxis-mshab", "mshab/runtime_bootstrap.py")
    module = _load_python_file("test_mshab_runtime_bootstrap", bootstrap_path)

    expected_asset_dir = tmp_path.resolve()
    monkeypatch.delenv("MS_ASSET_DIR", raising=False)

    fake_envs = types.SimpleNamespace()
    fake_make = types.SimpleNamespace(EnvConfig=object(), make_env=object())
    import_calls: list[tuple[str, str | None]] = []

    def fake_import_module(name: str):
        import_calls.append((name, os.environ.get("MS_ASSET_DIR")))
        if name == "mshab.envs":
            return fake_envs
        if name == "mshab.envs.make":
            return fake_make
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)

    resolved_asset_dir, env_config_cls, make_env_fn = module.load_env_factory(
        str(expected_asset_dir)
    )

    assert resolved_asset_dir == expected_asset_dir
    assert os.environ["MS_ASSET_DIR"] == str(expected_asset_dir)
    assert env_config_cls is fake_make.EnvConfig
    assert make_env_fn is fake_make.make_env
    assert import_calls == [
        ("mshab.envs", str(expected_asset_dir)),
        ("mshab.envs.make", str(expected_asset_dir)),
    ]


def test_mshab_builds_rgb_policy_observations_from_stacked_images() -> None:
    torch = pytest.importorskip("torch")
    praxis_eval = _load_mshab_praxis_eval_module()
    obs = {
        "state": torch.arange(84, dtype=torch.float32).reshape(2, 42),
        "pixels": {
            "fetch_head": torch.full((2, 3, 3, 128, 128), 128, dtype=torch.uint8),
            "fetch_hand": torch.full((2, 3, 128, 128), 0.25, dtype=torch.float32),
            "fetch_head_depth": torch.zeros((2, 3, 1, 128, 128), dtype=torch.float32),
        },
    }

    observations = praxis_eval.build_remote_observations(obs, policy_task="Pick")

    first = observations[0]
    assert set(first) == {
        "observation.state",
        "observation.images.fetch_head",
        "observation.images.fetch_hand",
        "task",
    }
    assert first["task"] == "Pick"
    assert first["observation.state"].shape == (42,)
    assert first["observation.images.fetch_head"].shape == (3, 128, 128)
    assert first["observation.images.fetch_head"].dtype == np.float32
    assert np.allclose(first["observation.images.fetch_head"], 128.0 / 255.0)
    assert np.allclose(first["observation.images.fetch_hand"], 0.25)
    assert "observation.images.fetch_head_depth" not in first


def test_mshab_accepts_unstacked_hwc_rgb_observations() -> None:
    praxis_eval = _load_mshab_praxis_eval_module()
    obs = {
        "state": np.zeros((1, 42), dtype=np.float32),
        "fetch_head": np.full((1, 128, 128, 3), 255, dtype=np.uint8),
        "fetch_hand": np.zeros((1, 128, 128, 3), dtype=np.uint8),
    }

    observations = praxis_eval.build_remote_observations(obs, policy_task="OpenFr")

    assert observations[0]["observation.images.fetch_head"].shape == (3, 128, 128)
    assert np.allclose(observations[0]["observation.images.fetch_head"], 1.0)
    assert np.allclose(observations[0]["observation.images.fetch_hand"], 0.0)


def test_mshab_rejects_wrong_rgb_resolution() -> None:
    praxis_eval = _load_mshab_praxis_eval_module()
    obs = {
        "state": np.zeros((1, 42), dtype=np.float32),
        "fetch_head": np.zeros((1, 64, 64, 3), dtype=np.uint8),
        "fetch_hand": np.zeros((1, 64, 64, 3), dtype=np.uint8),
    }

    with pytest.raises(ValueError, match="PolicyIO image shape"):
        praxis_eval.build_remote_observations(obs, policy_task="Pick")


def test_mshab_rejects_depth_only_observations() -> None:
    praxis_eval = _load_mshab_praxis_eval_module()
    obs = {
        "state": np.zeros((1, 42), dtype=np.float32),
        "pixels": {
            "fetch_head_depth": np.zeros((1, 1, 4, 5), dtype=np.float32),
            "fetch_hand_depth": np.zeros((1, 1, 4, 5), dtype=np.float32),
        },
    }

    with pytest.raises(KeyError, match="requires RGB keys"):
        praxis_eval.build_remote_observations(obs, policy_task="Pick")


def test_mshab_action_stats_track_bounds_and_shape_errors() -> None:
    praxis_eval = _load_mshab_praxis_eval_module()
    bounds = (
        np.array([-1.0, -0.5, -1.0], dtype=np.float32),
        np.array([1.0, 0.5, 1.0], dtype=np.float32),
    )
    stats = praxis_eval.ActionStatsAccumulator(action_dim=3, bounds=bounds)

    action = praxis_eval.prepare_policy_action(
        np.array([[0.0, 0.75, -2.0], [1.5, 0.0, 0.5]], dtype=np.float32),
        num_envs=2,
        action_dim=3,
    )
    stats.update(action)
    summary = stats.as_dict()

    assert summary["count"] == 2
    assert summary["min"] == [0.0, 0.0, -2.0]
    assert summary["max"] == [1.5, 0.75, 0.5]
    assert summary["outside_unit_count"] == [1, 0, 1]
    assert summary["row_fraction_any_outside_unit"] == 1.0
    assert summary["outside_bounds_count"] == [1, 1, 1]
    assert summary["row_fraction_any_outside_bounds"] == 1.0

    clipped_action = praxis_eval.clip_action_to_bounds(action, bounds)
    np.testing.assert_allclose(
        clipped_action,
        np.array([[0.0, 0.5, -1.0], [1.0, 0.0, 0.5]], dtype=np.float32),
    )
    clip_stats = praxis_eval.ActionClipStatsAccumulator(action_dim=3, enabled=True)
    clip_stats.update(action, clipped_action)
    clip_summary = clip_stats.as_dict()
    assert clip_summary["enabled"] is True
    assert clip_summary["clipped_count"] == [1, 1, 1]
    assert clip_summary["row_fraction_any_clipped"] == 1.0
    assert clip_summary["max_abs_delta"] == [0.5, 0.25, 1.0]

    with pytest.raises(ValueError, match="Expected policy action shape"):
        praxis_eval.prepare_policy_action(
            np.zeros((2, 4), dtype=np.float32),
            num_envs=2,
            action_dim=3,
        )
    with pytest.raises(ValueError, match="non-finite"):
        praxis_eval.prepare_policy_action(
            np.array([[np.nan, 0.0, 0.0]], dtype=np.float32),
            num_envs=1,
            action_dim=3,
        )


def test_mshab_summarizes_first_observation_batch() -> None:
    praxis_eval = _load_mshab_praxis_eval_module()
    observations = [
        {
            "observation.state": np.zeros((42,), dtype=np.float32),
            "observation.images.fetch_head": np.zeros((3, 128, 128), dtype=np.float32),
            "observation.images.fetch_hand": np.ones((3, 128, 128), dtype=np.float32),
            "task": "Pick",
        },
        {
            "observation.state": np.ones((42,), dtype=np.float32),
            "observation.images.fetch_head": np.ones((3, 128, 128), dtype=np.float32),
            "observation.images.fetch_hand": np.zeros((3, 128, 128), dtype=np.float32),
            "task": "Pick",
        },
    ]

    summary = praxis_eval.summarize_remote_observation_batch(observations)

    assert summary["batch_size"] == 2
    assert summary["tasks"] == ["Pick"]
    assert summary["observation.state"]["shape"] == [2, 42]
    assert summary["observation.images.fetch_head"]["shape"] == [2, 3, 128, 128]
    assert summary["observation.images.fetch_hand"]["mean"] == 0.5
