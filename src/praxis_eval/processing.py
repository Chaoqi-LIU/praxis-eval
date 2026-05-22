# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LeRobot-aligned processor pipeline wrappers for praxis-eval."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from lerobot.processor import PolicyAction, PolicyProcessorPipeline

IDENTITY_PROCESSOR_FACTORY = "identity"
EnvProcessorPair = tuple[Any, Any]


def make_policy_pre_post_processors() -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create default identity policy pre/post processors in LeRobot format."""
    from lerobot.processor import (
        IdentityProcessorStep,
        PolicyAction,
        PolicyProcessorPipeline,
    )
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[IdentityProcessorStep()],
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        name="PraxisEvalPolicyPreprocessor",
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[IdentityProcessorStep()],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        name="PraxisEvalPolicyPostprocessor",
    )
    return preprocessor, postprocessor


def make_identity_env_pre_post_processors(
    env_cfg: Any | None = None,
    policy_cfg: Any | None = None,
) -> EnvProcessorPair:
    """Create identity env processors."""
    from lerobot.processor import IdentityProcessorStep, PolicyProcessorPipeline

    _ = env_cfg, policy_cfg
    return (
        PolicyProcessorPipeline(steps=[IdentityProcessorStep()]),
        PolicyProcessorPipeline(steps=[IdentityProcessorStep()]),
    )


def make_env_pre_post_processors(env_cfg: Any, policy_cfg: Any) -> EnvProcessorPair:
    """Create environment processors from env-owned processor declarations."""
    processor_factory = getattr(env_cfg, "processor_factory", None)
    if processor_factory is not None:
        return _call_processor_factory(
            processor_factory,
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
        )

    from lerobot.envs.factory import (
        make_env_pre_post_processors as lerobot_make_env_pre_post_processors,
    )

    return cast(
        EnvProcessorPair,
        lerobot_make_env_pre_post_processors(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
        ),
    )


def _call_processor_factory(
    processor_factory: Any,
    *,
    env_cfg: Any,
    policy_cfg: Any,
) -> EnvProcessorPair:
    if processor_factory == IDENTITY_PROCESSOR_FACTORY:
        return make_identity_env_pre_post_processors(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
        )
    if callable(processor_factory):
        return cast(
            EnvProcessorPair,
            processor_factory(env_cfg=env_cfg, policy_cfg=policy_cfg),
        )
    if isinstance(processor_factory, str):
        module_name, separator, attr_name = processor_factory.partition(":")
        if separator != ":" or not module_name or not attr_name:
            raise ValueError(
                "processor_factory strings must use 'module:function' format; "
                f"got {processor_factory!r}."
            )
        factory = getattr(import_module(module_name), attr_name)
        return cast(EnvProcessorPair, factory(env_cfg=env_cfg, policy_cfg=policy_cfg))
    raise TypeError(
        "processor_factory must be 'identity', a callable, or a 'module:function' "
        f"string; got {type(processor_factory).__name__}."
    )
