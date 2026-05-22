from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from praxis_eval.scripts import _package_sources as sources


@dataclass
class _FakeDistribution:
    version: str = "1.2.3"
    direct_url: str | None = None

    def read_text(self, name: str) -> str | None:
        if name == "direct_url.json":
            return self.direct_url
        return None

    def locate_file(self, relative_path: str) -> Path:
        return Path(relative_path)


def test_runtime_setup_install_sources_use_editable_checkouts(tmp_path):
    root = tmp_path / "praxis-eval"
    remote = tmp_path / "praxis-remote"
    (remote / "src" / "praxis_remote").mkdir(parents=True)
    (remote / "pyproject.toml").write_text("[project]\nname = 'praxis-remote'\n")

    assert sources.praxis_eval_install(root) == ["-e", str(root)]
    assert sources.praxis_remote_install(root) == ["-e", str(remote)]


def test_runtime_setup_install_sources_use_installed_distribution_specs(monkeypatch):
    monkeypatch.setattr(
        sources,
        "distribution_install_args",
        lambda name, *, fallback=None: [f"{name}==1.2.3"],
    )

    assert sources.praxis_eval_install(None) == ["praxis-eval==1.2.3"]
    assert sources.praxis_remote_install(None) == ["praxis-remote==1.2.3"]


def test_distribution_install_args_preserves_vcs_direct_url(monkeypatch):
    direct_url = json.dumps(
        {
            "url": "https://github.com/Chaoqi-LIU/praxis-eval.git",
            "vcs_info": {"vcs": "git", "commit_id": "abc123"},
        }
    )
    monkeypatch.setattr(
        sources,
        "distribution",
        lambda _name: _FakeDistribution(direct_url=direct_url),
    )

    assert sources.distribution_install_args("praxis-eval") == [
        "praxis-eval @ git+https://github.com/Chaoqi-LIU/praxis-eval.git@abc123"
    ]


def test_distribution_install_args_preserves_editable_file_url(monkeypatch, tmp_path):
    checkout = tmp_path / "praxis-simpler"
    direct_url = json.dumps(
        {
            "url": checkout.as_uri(),
            "dir_info": {"editable": True},
        }
    )
    monkeypatch.setattr(
        sources,
        "distribution",
        lambda _name: _FakeDistribution(direct_url=direct_url),
    )

    assert sources.distribution_install_args("praxis-simpler") == [
        "-e",
        str(checkout.resolve()),
    ]


def test_distribution_install_args_preserves_editable_vcs_url(monkeypatch):
    direct_url = json.dumps(
        {
            "url": "https://github.com/Chaoqi-LIU/praxis-eval.git",
            "dir_info": {"editable": True},
            "vcs_info": {"vcs": "git", "commit_id": "abc123"},
        }
    )
    monkeypatch.setattr(
        sources,
        "distribution",
        lambda _name: _FakeDistribution(direct_url=direct_url),
    )

    assert sources.distribution_install_args("praxis-eval") == [
        "-e",
        "git+https://github.com/Chaoqi-LIU/praxis-eval.git@abc123#egg=praxis-eval",
    ]


def test_distribution_install_args_uses_pinned_version_without_direct_url(monkeypatch):
    monkeypatch.setattr(
        sources,
        "distribution",
        lambda _name: _FakeDistribution(version="0.1.0"),
    )

    assert sources.distribution_install_args("praxis-eval") == ["praxis-eval==0.1.0"]


def test_runtime_setup_fallback_constants_are_release_specs(monkeypatch):
    def missing_distribution(name: str):
        raise sources.PackageNotFoundError(name)

    monkeypatch.setattr(sources, "distribution", missing_distribution)

    assert sources.praxis_eval_install(None) == [sources.PRAXIS_EVAL_FALLBACK_INSTALL]
    assert sources.praxis_remote_install(None) == [
        sources.PRAXIS_REMOTE_FALLBACK_INSTALL
    ]
    assert sources.PRAXIS_EVAL_FALLBACK_INSTALL == "praxis-eval==0.1.1"
    assert sources.PRAXIS_REMOTE_FALLBACK_INSTALL == "praxis-remote>=0.1.0,<0.2.0"


def test_runtime_setup_reads_env_specs_from_published_distributions():
    simpler_env = sources.distribution_resource_path(
        "praxis-simpler",
        "simpler_env/praxis_conda_env.yaml",
    )
    mshab_env = sources.distribution_resource_path(
        "praxis-mshab",
        "mshab/praxis_conda_env.yaml",
    )

    assert simpler_env.read_text(encoding="utf-8").startswith("name: simpler-praxis")
    assert mshab_env.read_text(encoding="utf-8").startswith("name: mshab-praxis")
