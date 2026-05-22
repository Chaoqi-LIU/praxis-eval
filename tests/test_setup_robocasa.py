from __future__ import annotations

from praxis_eval.scripts import setup_robocasa


def test_default_dataset_base_path_uses_current_working_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("PRAXIS_EVAL_ROBOCASA_DATASET_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    assert (
        setup_robocasa.default_dataset_base_path()
        == (tmp_path / "data" / "robocasa").resolve()
    )


def test_default_dataset_base_path_uses_env_override(monkeypatch, tmp_path) -> None:
    root = tmp_path / "assets" / "robocasa"
    monkeypatch.setenv("PRAXIS_EVAL_ROBOCASA_DATASET_ROOT", str(root))

    assert setup_robocasa.default_dataset_base_path() == root.resolve()
