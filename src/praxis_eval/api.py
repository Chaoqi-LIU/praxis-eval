# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Top-level evaluation API."""

from __future__ import annotations

from praxis_eval.registry import EvalDriver, get_driver
from praxis_eval.types import EvalConfig, EvalResult, Policy


def evaluate(
    env: str | EvalDriver,
    *,
    policy: Policy,
    config: EvalConfig,
) -> EvalResult:
    """Evaluate ``policy`` with a registered env driver or explicit driver."""
    driver = get_driver(env) if isinstance(env, str) else env
    return driver.evaluate(policy=policy, config=config)
