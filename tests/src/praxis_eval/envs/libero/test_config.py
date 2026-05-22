from __future__ import annotations

import importlib
from types import SimpleNamespace

import yaml


def test_ensure_libero_config_writes_installed_package_paths(
    tmp_path, monkeypatch
) -> None:
    config_mod = importlib.import_module("praxis_eval.envs.libero.config")

    benchmark_root = tmp_path / "site-packages" / "libero" / "libero"
    benchmark_root.mkdir(parents=True)
    fake_origin = benchmark_root / "__init__.py"
    fake_origin.write_text("# stub\n", encoding="utf-8")

    monkeypatch.setattr(
        config_mod.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(fake_origin)),
    )

    config_root = tmp_path / "cfg"
    config_file = config_mod.ensure_libero_config(config_root=config_root)

    assert config_file == config_root / "config.yaml"
    assert config_file.exists()
    assert config_mod.os.environ["LIBERO_CONFIG_PATH"] == str(config_root.resolve())

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data == {
        "benchmark_root": str(benchmark_root.resolve()),
        "bddl_files": str((benchmark_root / "bddl_files").resolve()),
        "init_states": str((benchmark_root / "init_files").resolve()),
        "datasets": str((benchmark_root.parent / "datasets").resolve()),
        "assets": str((benchmark_root / "assets").resolve()),
    }


def test_libero_env_config_builds_camera_features() -> None:
    from lerobot.utils.constants import OBS_IMAGES

    from praxis_eval.envs.libero.config import LiberoEnvConfig

    cfg = LiberoEnvConfig(
        observation_height=96,
        observation_width=128,
        camera_name="agentview_image,robot0_eye_in_hand_image",
    )

    assert cfg.features["action"].shape == (7,)
    assert cfg.features["pixels/agentview_image"].shape == (96, 128, 3)
    assert cfg.features_map["pixels/agentview_image"] == f"{OBS_IMAGES}.image"
    assert cfg.features_map["pixels/robot0_eye_in_hand_image"] == (
        f"{OBS_IMAGES}.image2"
    )
