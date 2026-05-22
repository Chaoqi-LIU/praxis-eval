from __future__ import annotations

import numpy as np

from praxis_eval import (
    ActionSpec,
    EnvContract,
    EvalConfig,
    EvalPhaseWatchdog,
    EvalResult,
    LocalPolicy,
    available_drivers,
    evaluate,
    get_driver,
    normalize_batched_action,
    register_driver,
    require_positive_int,
)
from praxis_eval.contracts import ObservationKey
from praxis_eval.metrics import EpisodeResult, summarize_episodes


class OneStepDriver:
    @property
    def contract(self) -> EnvContract:
        return EnvContract(
            env_type="one_step",
            observation_keys=(ObservationKey("task", "str"),),
            action=ActionSpec(shape=(2,), dtype="float32", minimum=-1.0, maximum=1.0),
        )

    def evaluate(self, *, policy, config: EvalConfig) -> EvalResult:
        policy.reset()
        action = policy.act(
            [{"task": config.task or "test"}],
            action_spec=self.contract.action,
            policy_kwargs=config.policy_kwargs,
        )
        self.contract.action.validate(action[0])
        summary = summarize_episodes(
            [EpisodeResult(task_key="one_step/0", success=True, episode_length=1)]
        )
        return EvalResult(
            overall=summary["overall"],
            per_task=summary["per_task"],
            metadata={"action": action.tolist()},
        )


def test_evaluate_dispatches_registered_driver() -> None:
    register_driver("one_step", OneStepDriver())

    def policy(observations, *, policy_kwargs=None, episode_ids=None):
        assert observations == [{"task": "pick"}]
        assert policy_kwargs == {"decode_keep_k": 2}
        assert episode_ids is None
        return np.asarray([[0.0, 1.0]], dtype=np.float32)

    result = evaluate(
        "one_step",
        policy=LocalPolicy(policy),
        config=EvalConfig(
            task="pick",
            num_eval_per_task=1,
            output_dir="unused",
            policy_kwargs={"decode_keep_k": 2},
        ),
    )

    assert result.overall["success_rate"] == 1.0
    assert result.metadata["action"] == [[0.0, 1.0]]


def test_local_policy_validates_batched_action_shape() -> None:
    policy = LocalPolicy(lambda observations, **_: np.zeros((1, 2), dtype=np.float32))

    action = policy.act(
        [{"task": "test"}],
        action_spec=ActionSpec(shape=(2,), dtype="float32"),
    )

    np.testing.assert_array_equal(action, np.zeros((1, 2), dtype=np.float32))


def test_local_policy_normalizes_single_unbatched_action() -> None:
    policy = LocalPolicy(lambda observations, **_: np.zeros((2,), dtype=np.float32))

    action = policy.act([{"task": "test"}])

    np.testing.assert_array_equal(action, np.zeros((1, 2), dtype=np.float32))


def test_local_policy_rejects_wrong_batch_dim_without_action_spec() -> None:
    policy = LocalPolicy(lambda observations, **_: np.zeros((2, 2), dtype=np.float32))

    with np.testing.assert_raises(ValueError):
        policy.act([{"task": "test"}])


def test_local_policy_rejects_empty_observation_batch_before_calling_policy() -> None:
    called = False

    def policy(observations, *, policy_kwargs=None, episode_ids=None):
        nonlocal called
        called = True
        return np.zeros((0, 2), dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        LocalPolicy(policy).act([])

    assert called is False


def test_builtin_contract_drivers_are_discoverable() -> None:
    assert "libero" in available_drivers()
    assert get_driver("libero").contract.env_type == "libero"


def test_top_level_exports_integration_helpers() -> None:
    assert require_positive_int(1, name="value") == 1
    assert EvalPhaseWatchdog.__name__ == "EvalPhaseWatchdog"
    np.testing.assert_array_equal(
        normalize_batched_action(
            np.zeros((2,), dtype=np.float32),
            batch_size=1,
            action_spec=None,
        ),
        np.zeros((1, 2), dtype=np.float32),
    )


def test_builtin_contract_registration_does_not_overwrite_real_driver(
    monkeypatch,
) -> None:
    import praxis_eval.registry as registry

    driver = OneStepDriver()
    monkeypatch.setattr(registry, "_DRIVERS", {}, raising=False)
    monkeypatch.setattr(registry, "_BUILTINS_REGISTERED", False, raising=False)
    registry.register_driver("libero", driver)

    assert registry.get_driver("libero") is driver
