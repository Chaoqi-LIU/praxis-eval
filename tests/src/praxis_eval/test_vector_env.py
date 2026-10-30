"""Tests for the RL vector env over eval pools."""

from __future__ import annotations

from functools import partial

import gymnasium as gym
import numpy as np
from gymnasium.vector.vector_env import AutoresetMode
from gymnasium.wrappers import TimeLimit

from praxis_eval.contracts import ActionSpec
from praxis_eval.envs.async_vector_env import make_eval_env_pool
from praxis_eval.envs.eval_pool import EvalLaneJob, EvalPoolHandle
from praxis_eval.evaluation.sim.adapters import LocalPolicyAdapter
from praxis_eval.evaluation.sim.rollout_compat import evaluate_policy_on_pooled_env
from praxis_eval.processing import (
    make_identity_env_pre_post_processors,
    make_policy_pre_post_processors,
)
from praxis_eval.vector_env import EvalVectorEnv


class _LaneEnv(gym.Env):
    """LIBERO-like lane: never truncates, and task "a" succeeds at step 2.

    ``agent_pos`` records ``(reset seed, episode_index, step, last action)``;
    ``resets`` and ``steps`` count calls.
    """

    def __init__(self, job: EvalLaneJob) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
                "pixels": gym.spaces.Box(0, 255, (8, 8, 3), np.uint8),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.task_description = f"{job.task_group} {job.task_id}"
        self.episode_index = int(job.episode_index)
        self._success_at = 2 if job.task_group == "a" else None
        self._max_episode_steps = 10 if job.task_group == "a" else 3
        self._seed = 0
        self._t = 0
        self.resets = 0
        self.steps = 0

    def prepare_eval_job(self, task_id, episode_index, task_group=None) -> None:
        self.episode_index = int(episode_index)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self._seed, self._t = (0 if seed is None else int(seed)), 0
        return self._obs(0.0), {}

    def step(self, action):
        self.steps += 1
        self._t += 1
        success = self._success_at is not None and self._t >= self._success_at
        info = {"is_success": success}
        if success:
            info["final_info"] = {"is_success": True}
        return self._obs(float(action[0])), float(success), success, False, info

    def _obs(self, last_action: float) -> dict[str, np.ndarray]:
        state = [self._seed, self.episode_index, self._t, last_action]
        pixels = np.random.default_rng(self._seed + self._t).integers(
            0, 256, (8, 8, 3), dtype=np.uint8
        )
        return {"agent_pos": np.asarray(state, np.float32), "pixels": pixels}


def _make_eval_pool(num_envs: int, prepared: list[list[EvalLaneJob]]):
    """Pool shaped like the builtin ones: first prepare builds, later ones call_each."""
    handle: EvalPoolHandle

    def _prepare_jobs(lane_jobs):
        prepared.append(list(lane_jobs))
        if handle.env_pool is None:
            handle.env_pool = make_eval_env_pool(
                [partial(_LaneEnv, j) for j in lane_jobs]
            )
            return
        handle.env_pool.call_each(
            "prepare_eval_job",
            args_list=[(job.task_id, job.episode_index) for job in lane_jobs],
            kwargs_list=[{"task_group": job.task_group} for job in lane_jobs],
        )

    handle = EvalPoolHandle(
        env_pool=None, num_envs=num_envs, prepare_jobs=_prepare_jobs
    )
    return handle


def _make_vector_env(lanes, **kwargs):
    prepared: list[list[EvalLaneJob]] = []
    env_preprocessor, env_postprocessor = make_identity_env_pre_post_processors()
    env = EvalVectorEnv(
        _make_eval_pool(len(lanes), prepared),
        lanes=lanes,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        action_spec=ActionSpec(shape=(2,), dtype="float32", minimum=-1, maximum=1),
        **kwargs,
    )
    return env, prepared


def _reference_env(lanes) -> gym.vector.SyncVectorEnv:
    """Gymnasium's own SAME_STEP vector env over the same lane envs."""

    def _make(task):
        lane = _LaneEnv(EvalLaneJob(task[0], task[1], 0, 0))
        return TimeLimit(lane, max_episode_steps=lane._max_episode_steps)

    return gym.vector.SyncVectorEnv(
        [partial(_make, task) for task in lanes],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


def _step_like_reference(env, reference, actions):
    """Step both envs and check that everything but the observations agrees."""
    result = env.step(actions)
    _, rewards, terminated, truncated, info = reference.step(actions)
    np.testing.assert_array_equal(result[1], rewards)
    np.testing.assert_array_equal(result[2], terminated)
    np.testing.assert_array_equal(result[3], truncated)
    _assert_same_info(result[4], info)
    return result


def _assert_same_info(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, dict):
            _assert_same_info(actual[key], value)
        elif key == "final_obs":
            # Raw vs processed observations differ, but not which lanes have one.
            assert actual[key].dtype == object
            assert [x is None for x in actual[key]] == [x is None for x in value]
        else:
            assert actual[key].dtype == value.dtype, key
            np.testing.assert_array_equal(actual[key], value)


def _actions(*first: float) -> np.ndarray:
    return np.asarray([[value, 0.0] for value in first], dtype=np.float32)


class _RecordingPolicy:
    def __init__(self) -> None:
        self.observations: list[list[dict]] = []

    def reset(self, episode_ids=None) -> None:
        pass

    def act(
        self, observations, *, action_spec=None, policy_kwargs=None, episode_ids=None
    ):
        self.observations.append(list(observations))
        return np.full((len(observations), 2), 0.5, dtype=np.float32)


def test_done_lanes_restart_within_the_same_step() -> None:
    lanes = [("a", 0), ("a", 0), ("b", 1)]
    env, _ = _make_vector_env(lanes, seed=10)
    reference = _reference_env(lanes)
    try:
        obs, _ = env.reset()
        reference.reset()
        np.testing.assert_allclose(
            obs["observation.state"], [[10, 0, 0, 0], [12, 2, 0, 0], [11, 1, 0, 0]]
        )

        *_, info = _step_like_reference(env, reference, _actions(0.1, 0.2, 0.3))
        assert "final_obs" not in info

        # Task "a" succeeds at step 2: both "a" lanes terminate and restart.
        obs, _, terminated, _, info = _step_like_reference(
            env, reference, _actions(0.1, 0.2, 0.3)
        )
        assert terminated.tolist() == [True, True, False]
        assert info["_final_info"].tolist() == [True, True, False]
        assert info["final_info"]["is_success"].tolist() == [True, True, False]
        assert info["_final_obs"].tolist() == [True, True, False]
        final_obs = info["final_obs"]
        np.testing.assert_allclose(final_obs[0]["observation.state"], [10, 0, 2, 0.1])
        np.testing.assert_allclose(final_obs[1]["observation.state"], [12, 2, 2, 0.2])
        assert (final_obs[0]["task"], final_obs[1]["task"]) == ("a 0", "a 0")
        assert final_obs[2] is None
        np.testing.assert_allclose(
            obs["observation.state"],
            [[14, 4, 0, 0], [16, 6, 0, 0], [11, 1, 2, 0.3]],
            atol=1e-6,
        )

        # Lane 2 reaches its 3-step horizon. The restarted lanes took exactly one
        # step with this action, so no pending autoreset swallowed it.
        obs, _, _, truncated, info = _step_like_reference(
            env, reference, _actions(0.4, 0.5, 0.6)
        )
        assert truncated.tolist() == [False, False, True]
        assert info["_final_info"].tolist() == [False, False, True]
        assert info["final_info"]["_is_success"].tolist() == [False, False, True]
        assert not info["final_info"]["is_success"][2]
        np.testing.assert_allclose(
            info["final_obs"][2]["observation.state"], [11, 1, 3, 0.6]
        )
        np.testing.assert_allclose(
            obs["observation.state"],
            [[14, 4, 1, 0.4], [16, 6, 1, 0.5], [13, 3, 0, 0]],
            atol=1e-6,
        )
    finally:
        env.close()
        reference.close()


def test_observation_space_describes_observations() -> None:
    env, _ = _make_vector_env([("a", 0), ("b", 1)], episode_length=1)
    try:
        obs, _ = env.reset()
        arrays = {key: obs[key] for key in env.observation_space.spaces}
        assert env.observation_space.contains(arrays)
        for key, space in env.observation_space.spaces.items():
            assert arrays[key].dtype == space.dtype
        # The keys outside the space are the documented non-array values.
        assert {key: obs[key] for key in obs.keys() - arrays.keys()} == {
            "task": ["a 0", "b 1"],
            "next.reward": 0.0,
            "next.done": False,
            "next.truncated": False,
        }

        *_, info = env.step(_actions(0.0, 0.0))
        for final_obs in info["final_obs"]:
            lane_arrays = {key: final_obs[key] for key in arrays}
            assert env.single_observation_space.contains(lane_arrays)
    finally:
        env.close()


def test_lanes_number_episodes_per_task_like_the_eval_runner() -> None:
    env, prepared = _make_vector_env(
        [("a", 0), ("b", 1), ("a", 0)], seed=100, episode_length=1
    )
    try:
        env.reset()
        env.step(_actions(0.0, 0.0, 0.0))  # every lane truncates and restarts
        first = [EvalLaneJob("a", 0, 0, 0), EvalLaneJob("b", 1, 0, 1)]
        first.append(EvalLaneJob("a", 0, 1, 2))
        second = [EvalLaneJob("a", 0, 2, 4), EvalLaneJob("b", 1, 1, 3)]
        second.append(EvalLaneJob("a", 0, 3, 6))
        # Pool build, reset(), then the restart inside step().
        assert prepared == [first, first, second]
        assert env.lane_jobs == tuple(second)

        obs, _ = env.reset(seed=7)  # restarts the per-task counters
        assert prepared[-1] == first
        np.testing.assert_allclose(
            obs["observation.state"][:, :2], [[7, 0], [8, 1], [9, 2]]
        )
    finally:
        env.close()


def test_observations_match_what_evaluate_feeds_policies() -> None:
    lanes = [("a", 0), ("b", 1)]
    jobs = [EvalLaneJob("a", 0, 0, 0), EvalLaneJob("b", 1, 0, 1)]
    eval_pool = _make_eval_pool(2, [])
    eval_pool.prepare_jobs(jobs)
    policy = _RecordingPolicy()
    env_preprocessor, env_postprocessor = make_identity_env_pre_post_processors()
    preprocessor, postprocessor = make_policy_pre_post_processors()
    try:
        evaluate_policy_on_pooled_env(
            env=eval_pool.env_pool,
            policy=LocalPolicyAdapter(policy, device="cpu"),
            seeds=[5 + job.episode_index for job in jobs],
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
        )
    finally:
        eval_pool.close(terminate=True)

    env, _ = _make_vector_env(lanes, seed=5)
    try:
        batches = [env.reset()[0], env.step(_actions(0.5, 0.5))[0]]
    finally:
        env.close()

    for batch, policy_observations in zip(batches, policy.observations, strict=False):
        for lane, expected in enumerate(policy_observations):
            assert batch.keys() == expected.keys()
            for key, value in expected.items():
                lane_value = batch[key]
                if isinstance(lane_value, (np.ndarray, list)):
                    lane_value = lane_value[lane]
                assert type(lane_value) is type(value)
                np.testing.assert_array_equal(lane_value, value)
                assert np.asarray(lane_value).dtype == np.asarray(value).dtype


def test_finished_eval_lanes_wait_without_stepping_or_resetting() -> None:
    jobs = [EvalLaneJob("a", 0, 0, 0), EvalLaneJob("b", 1, 0, 1)]
    env_pool = make_eval_env_pool([partial(_LaneEnv, job) for job in jobs])
    try:
        # Lane "a" succeeds at step 2, then waits while lane "b" runs to step 5.
        env_pool.set_attr("_max_episode_steps", [10, 5])
        results = evaluate_policy_on_pooled_env(
            env=env_pool,
            policy=LocalPolicyAdapter(_RecordingPolicy(), device="cpu"),
            seeds=[0, 1],
        )
        resets = env_pool.call("resets")
        steps = env_pool.call("steps")
    finally:
        env_pool.close(terminate=True)

    assert [result.successes for result in results] == [[True], [False]]
    assert [result.lengths for result in results] == [[2], [5]]
    assert resets == (1, 1)
    assert steps == (2, 5)
