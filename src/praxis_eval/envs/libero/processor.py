# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO environment processor wiring."""

from __future__ import annotations

from typing import Any

from lerobot.processor import IdentityProcessorStep, PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep


def make_libero_env_pre_post_processors(
    env_cfg: Any,
    policy_cfg: Any,
):
    """Create LIBERO env processors using LeRobot's canonical processor step."""
    _ = env_cfg, policy_cfg
    return (
        PolicyProcessorPipeline(steps=[LiberoProcessorStep()]),
        PolicyProcessorPipeline(steps=[IdentityProcessorStep()]),
    )
