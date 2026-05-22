from __future__ import annotations

from pathlib import Path

from praxis_eval.scripts import _verify_common


def test_resolve_output_dir_uses_explicit_path(tmp_path: Path) -> None:
    out = _verify_common.resolve_output_dir(tmp_path / "custom", "ignored")

    assert out == (tmp_path / "custom").resolve()


def test_resolve_output_dir_uses_default_name() -> None:
    out = _verify_common.resolve_output_dir(None, "robocasa")

    assert out.parent.name == "robocasa"
    assert out.parent.parent.name == "praxis_eval_verify"


def test_resolve_env_python_bin_uses_explicit_path(tmp_path: Path) -> None:
    python_bin = tmp_path / "env" / "bin" / "python"

    assert _verify_common.resolve_env_python_bin(python_bin, "unused") == (
        python_bin.resolve()
    )


def test_resolve_env_python_bin_falls_back_to_named_env(monkeypatch) -> None:
    resolved = Path("/tmp/test-env/bin/python")
    monkeypatch.setattr(
        _verify_common,
        "resolve_named_env_python",
        lambda env_name: resolved if env_name == "named-env" else None,
    )

    assert _verify_common.resolve_env_python_bin(None, "named-env") == resolved
