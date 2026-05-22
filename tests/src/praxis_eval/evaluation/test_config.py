"""Tests for shared eval config parsing helpers."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from praxis_eval.evaluation.config import (
    require_positive_int,
    resolve_nonnegative_int,
    resolve_optional_positive_timeout_sec,
    resolve_policy_kwargs,
)


def test_require_positive_int() -> None:
    assert require_positive_int(3, name="x") == 3
    with pytest.raises(ValueError, match="x must be >= 1"):
        require_positive_int(0, name="x")


def test_resolve_nonnegative_int() -> None:
    cfg = OmegaConf.create({"rollout_failure_retries": "2"})
    assert (
        resolve_nonnegative_int(
            cfg,
            key="rollout_failure_retries",
            label="eval.rollout_failure_retries",
            default=1,
        )
        == 2
    )
    assert (
        resolve_nonnegative_int(
            {},
            key="rollout_failure_retries",
            label="eval.rollout_failure_retries",
            default=1,
        )
        == 1
    )
    with pytest.raises(ValueError, match="eval.rollout_failure_retries must be >= 0"):
        resolve_nonnegative_int(
            {"rollout_failure_retries": -1},
            key="rollout_failure_retries",
            label="eval.rollout_failure_retries",
        )


@pytest.mark.parametrize("raw", [None, 0, -1])
def test_resolve_optional_positive_timeout_disables_null_or_nonpositive(raw) -> None:
    cfg = OmegaConf.create({"step_timeout_sec": raw})
    assert resolve_optional_positive_timeout_sec(cfg) is None


def test_resolve_optional_positive_timeout_returns_positive_float() -> None:
    cfg = OmegaConf.create({"step_timeout_sec": "12.5"})
    assert resolve_optional_positive_timeout_sec(cfg) == 12.5


def test_resolve_policy_kwargs_filters_none_values() -> None:
    cfg = OmegaConf.create(
        {
            "policy_kwargs": {
                "decode_keep_k": 4,
                "unused": None,
            }
        }
    )
    assert resolve_policy_kwargs(cfg, label="eval.policy_kwargs") == {
        "decode_keep_k": 4,
    }


def test_resolve_policy_kwargs_rejects_non_mapping() -> None:
    cfg = OmegaConf.create({"policy_kwargs": ["bad"]})
    with pytest.raises(TypeError, match="eval.policy_kwargs must resolve to a dict"):
        resolve_policy_kwargs(cfg, label="eval.policy_kwargs")
