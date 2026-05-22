"""Tests for evaluation override helpers."""

from __future__ import annotations

import subprocess
import sys

import pytest

from praxis_eval.evaluation.overrides import (
    build_eval_overrides_from_train_config,
    normalize_eval_overrides,
)


def test_normalize_eval_overrides_adds_plus_for_dynamic_policy_kwargs():
    assert normalize_eval_overrides(
        [
            "eval.policy_kwargs.decode_keep_k=4",
            "eval.policy_kwargs.custom_flag=4",
            "+eval.policy_kwargs.already_dynamic=true",
        ]
    ) == [
        "eval.policy_kwargs.decode_keep_k=4",
        "+eval.policy_kwargs.custom_flag=4",
        "+eval.policy_kwargs.already_dynamic=true",
    ]


def test_build_eval_overrides_resolves_train_config_interpolations(tmp_path):
    train_config = tmp_path / "train_config.yaml"
    train_config.write_text(
        "\n".join(
            [
                "device_name: cuda",
                "policy:",
                "  name: bar_boat_pali",
                "dataset:",
                "  name: bridge_single_view",
                "device: ${device_name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    overrides = build_eval_overrides_from_train_config(
        train_config,
        num_eval_per_task=10,
        num_parallel_env=4,
        record_episodes_per_task=2,
    )

    assert "policy=bar_boat_pali" in overrides
    assert "env.type=simpler" in overrides
    assert "env.task=bridge" in overrides
    assert "device=cuda" in overrides


def test_build_eval_overrides_infers_mshab_clean_subset_from_dataset_name(tmp_path):
    train_config = tmp_path / "train_config.yaml"
    train_config.write_text(
        "\n".join(
            [
                "policy:",
                "  name: bar_oat_pali",
                "dataset:",
                "  name: mshab_settable",
                "  repo_id: chaoqi-liu/mshab_settable_clean",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    overrides = build_eval_overrides_from_train_config(
        train_config,
        num_eval_per_task=50,
        num_parallel_env=10,
        record_episodes_per_task=1,
    )

    assert "policy=bar_oat_pali" in overrides
    assert "env.type=mshab" in overrides
    assert "env.task='pick,place'" in overrides
    assert "eval.num_eval_per_task=50" in overrides
    assert "eval.num_parallel_env=10" in overrides
    assert "eval.record_episodes_per_task=1" in overrides


def test_build_eval_overrides_infers_metaworld_mt50_from_dataset_name(tmp_path):
    train_config = tmp_path / "train_config.yaml"
    train_config.write_text(
        "\n".join(
            [
                "policy:",
                "  name: bar_oat_pali",
                "dataset:",
                "  name: metaworld_mt50",
                "  repo_id: lerobot/metaworld_mt50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    overrides = build_eval_overrides_from_train_config(
        train_config,
        num_eval_per_task=50,
        num_parallel_env=50,
        record_episodes_per_task=1,
    )

    assert "policy=bar_oat_pali" in overrides
    assert "env.type=metaworld" in overrides
    assert "env.task=mt50" in overrides


def test_importing_override_helpers_does_not_import_sim_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import praxis_eval.evaluation.overrides; "
                "assert 'praxis_eval.evaluation.sim' not in sys.modules"
            ),
        ],
        check=True,
        timeout=5,
    )


def test_build_eval_overrides_ignores_unrelated_unsupported_interpolations(tmp_path):
    train_config = tmp_path / "train_config.yaml"
    train_config.write_text(
        "\n".join(
            [
                "run_name: ${now:%H%M%S}_train",
                "policy:",
                "  name: bar_boat_pali",
                "env:",
                "  type: libero",
                "  task: libero_10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    overrides = build_eval_overrides_from_train_config(
        train_config,
        num_eval_per_task=10,
        num_parallel_env=4,
        record_episodes_per_task=2,
    )

    assert "policy=bar_boat_pali" in overrides
    assert "env.type=libero" in overrides
    assert "env.task=libero_10" in overrides


def test_build_eval_overrides_quotes_comma_tasks_and_preserves_task_ids(tmp_path):
    train_config = tmp_path / "train_config.yaml"
    train_config.write_text(
        "\n".join(
            [
                "policy:",
                "  name: bar_fast_pali",
                "env:",
                "  type: libero",
                "  task: libero_goal,libero_10",
                "  task_ids: [1, 3]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    overrides = build_eval_overrides_from_train_config(
        train_config,
        num_eval_per_task=8,
        num_parallel_env=2,
        record_episodes_per_task=0,
    )

    assert "env.task='libero_goal,libero_10'" in overrides
    assert "env.task_ids=[1, 3]" in overrides
    assert "eval.num_eval_per_task=8" in overrides
    assert "eval.num_parallel_env=2" in overrides
    assert "eval.record_episodes_per_task=0" in overrides


def test_build_eval_overrides_accepts_custom_dataset_inferer(tmp_path):
    train_config = tmp_path / "train_config.yaml"
    train_config.write_text(
        "\n".join(
            [
                "policy:",
                "  name: custom_policy",
                "dataset:",
                "  name: custom_dataset",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    overrides = build_eval_overrides_from_train_config(
        train_config,
        num_eval_per_task=1,
        num_parallel_env=1,
        record_episodes_per_task=0,
        infer_env_from_dataset=lambda name: ("custom_env", f"{name}_task"),
    )

    assert "env.type=custom_env" in overrides
    assert "env.task=custom_dataset_task" in overrides


def test_build_eval_overrides_requires_policy_name(tmp_path):
    train_config = tmp_path / "train_config.yaml"
    train_config.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: bridge_single_view",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="policy.name"):
        build_eval_overrides_from_train_config(
            train_config,
            num_eval_per_task=1,
            num_parallel_env=1,
            record_episodes_per_task=0,
        )
