"""Tests for evaluation artifact helpers."""

from __future__ import annotations

import json

from praxis_eval.evaluation.artifacts import (
    resolve_eval_artifact_paths,
    resolve_eval_step_dir,
    write_eval_results_json,
)


class TestEvalArtifacts:
    def test_resolve_eval_artifact_paths_creates_dirs(self, tmp_path):
        out_dir, media_dir = resolve_eval_artifact_paths(
            tmp_path / "eval_out", media_dirname="media"
        )
        assert out_dir.exists()
        assert media_dir.exists()
        assert media_dir == out_dir / "media"

    def test_write_eval_results_json(self, tmp_path):
        results = {
            "overall": {"success_rate": 75.0},
            "per_group": {"suite": {"success_rate": 75.0}},
            "per_task": {"suite/0": {"success_rate": 75.0}},
        }
        out_file = write_eval_results_json(
            results=results,
            output_dir=tmp_path / "eval_out",
            results_filename="results.json",
            metadata={"seed": 42},
        )
        payload = json.loads(out_file.read_text())
        assert payload["overall"]["success_rate"] == 75.0
        assert payload["_meta"]["seed"] == 42
        assert "timestamp_utc" in payload["_meta"]

    def test_resolve_eval_step_dir_uses_by_step_layout(self, tmp_path):
        step_dir = resolve_eval_step_dir(tmp_path / "eval", 9500)
        assert step_dir == tmp_path / "eval" / "by_step" / "step_9500"
