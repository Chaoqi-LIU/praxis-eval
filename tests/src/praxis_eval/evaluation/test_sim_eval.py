"""Tests for simulation evaluation runtime wrappers."""

from __future__ import annotations

import multiprocessing
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from praxis_eval.contracts import ActionSpec
from praxis_eval.envs.eval_pool import EvalPoolHandle
from praxis_eval.evaluation.sim import (
    LocalPolicyAdapter,
    evaluate_policy_on_env_pool,
    rollout_compat,
)
from praxis_eval.evaluation.sim import (
    runner as sim_runner,
)
from praxis_eval.evaluation.sim import (
    wave_retry as sim_wave_retry,
)
from praxis_eval.evaluation.sim.diagnostics import (
    InferenceDiagnosticsAccumulator,
)
from praxis_eval.evaluation.sim.metrics import TaskEvalResult


class _TinyEnv(gym.Env):
    metadata = {"render_fps": 30}

    def __init__(
        self, success_at: int = 2, max_steps: int = 5, task_desc: str = "tiny task"
    ):
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
                "pixels": gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
            }
        )
        self._success_at = success_at
        self._max_episode_steps = max_steps
        self._task_desc = task_desc
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        obs = {
            "agent_pos": np.zeros(2, dtype=np.float32),
            "pixels": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        return obs, {}

    def step(self, action):
        self._step += 1
        success = self._step >= self._success_at
        terminated = success or self._step >= self._max_episode_steps
        obs = {
            "agent_pos": np.zeros(2, dtype=np.float32),
            "pixels": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        reward = 1.0 if success else 0.0
        return obs, reward, terminated, False, {"is_success": success}

    def task_description(self) -> str:
        return self._task_desc

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _RecordingTinyEnv(_TinyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions: list[np.ndarray] = []

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        return super().step(action)


class _TinyPolicy:
    def __init__(self):
        self.last_kwargs: dict[str, object] | None = None
        self._inference_info: list[dict[str, Any]] = []

    def reset(self, episode_ids=None) -> None:
        _ = episode_ids

    def act(
        self,
        observations,
        *,
        action_spec=None,
        policy_kwargs=None,
        episode_ids=None,
    ) -> np.ndarray:
        _ = (action_spec, episode_ids)
        self.last_kwargs = dict(policy_kwargs or {})
        return np.zeros((len(observations), 2), dtype=np.float32)

    def _record_inference_info(self, info: dict[str, Any]) -> None:
        self._inference_info.append(dict(info))

    def consume_inference_info(self) -> list[dict[str, Any]]:
        info = list(self._inference_info)
        self._inference_info.clear()
        return info


class _RecordingPolicy(_TinyPolicy):
    def __init__(self):
        super().__init__()
        self.observations = None
        self.action_spec = None

    def act(
        self,
        observations,
        *,
        action_spec=None,
        policy_kwargs=None,
        episode_ids=None,
    ) -> np.ndarray:
        self.observations = list(observations)
        self.action_spec = action_spec
        return super().act(
            observations,
            action_spec=action_spec,
            policy_kwargs=policy_kwargs,
            episode_ids=episode_ids,
        )


class _BadActionPolicy(_TinyPolicy):
    def act(
        self,
        observations,
        *,
        action_spec=None,
        policy_kwargs=None,
        episode_ids=None,
    ) -> np.ndarray:
        _ = (action_spec, episode_ids)
        self.last_kwargs = dict(policy_kwargs or {})
        action = np.array([float("nan"), 2.5], dtype=np.float32)
        return np.repeat(action[None, :], len(observations), axis=0)


class _OutOfBoundsActionPolicy(_TinyPolicy):
    def __init__(self):
        super().__init__()
        self.action_spec = None

    def act(
        self,
        observations,
        *,
        action_spec=None,
        policy_kwargs=None,
        episode_ids=None,
    ) -> np.ndarray:
        _ = episode_ids
        self.action_spec = action_spec
        self.last_kwargs = dict(policy_kwargs or {})
        action = np.array([1.25, -1.25], dtype=np.float32)
        return np.repeat(action[None, :], len(observations), axis=0)


class _DiagnosticTinyPolicy(_TinyPolicy):
    def act(
        self,
        observations,
        *,
        action_spec=None,
        policy_kwargs=None,
        episode_ids=None,
    ) -> np.ndarray:
        action = super().act(
            observations,
            action_spec=action_spec,
            policy_kwargs=policy_kwargs,
            episode_ids=episode_ids,
        )
        self._record_inference_info(
            {
                "decode_keep_k": [4, 8],
                "effective_decode_keep_k": [4, 8],
                "decode_steps": 2,
            }
        )
        return action


class _TinyEnvNoTask(gym.Env):
    metadata = {"render_fps": 30}

    def __init__(self, success_at: int = 2, max_steps: int = 5):
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
                "pixels": gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
            }
        )
        self._success_at = success_at
        self._max_episode_steps = max_steps
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        obs = {
            "agent_pos": np.zeros(2, dtype=np.float32),
            "pixels": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        return obs, {}

    def step(self, action):
        self._step += 1
        success = self._step >= self._success_at
        terminated = success or self._step >= self._max_episode_steps
        obs = {
            "agent_pos": np.zeros(2, dtype=np.float32),
            "pixels": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        reward = 1.0 if success else 0.0
        return obs, reward, terminated, False, {"is_success": success}

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _TinyNestedPixelsEnv(gym.Env):
    metadata = {"render_fps": 30}

    def __init__(
        self, success_at: int = 2, max_steps: int = 5, task_desc: str = "tiny task"
    ):
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
                "pixels": gym.spaces.Dict(
                    {
                        "cam0": gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
                        "cam1": gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
                    }
                ),
            }
        )
        self._success_at = success_at
        self._max_episode_steps = max_steps
        self._task_desc = task_desc
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        obs = {
            "agent_pos": np.zeros(2, dtype=np.float32),
            "pixels": {
                "cam0": np.zeros((8, 8, 3), dtype=np.uint8),
                "cam1": np.zeros((8, 8, 3), dtype=np.uint8),
            },
        }
        return obs, {}

    def step(self, action):
        self._step += 1
        success = self._step >= self._success_at
        terminated = success or self._step >= self._max_episode_steps
        obs = {
            "agent_pos": np.zeros(2, dtype=np.float32),
            "pixels": {
                "cam0": np.zeros((8, 8, 3), dtype=np.uint8),
                "cam1": np.zeros((8, 8, 3), dtype=np.uint8),
            },
        }
        reward = 1.0 if success else 0.0
        return obs, reward, terminated, False, {"is_success": success}

    def task_description(self) -> str:
        return self._task_desc

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _FakePooledEnv:
    def __init__(self, num_envs: int):
        self.num_envs = int(num_envs)
        self.prepared_waves: list[list[tuple[int, int]]] = []
        self.closed = False

    def call_each(self, name: str, *, args_list, kwargs_list=None, timeout=None):
        _ = (timeout,)
        assert name == "prepare_eval_job"
        wave = [
            (int(task_id), int(episode_index)) for task_id, episode_index in args_list
        ]
        assert len(wave) == self.num_envs
        self.prepared_waves.append(wave)
        return tuple([None] * self.num_envs)

    def close(self, *args, **kwargs):
        _ = (args, kwargs)
        self.closed = True


def _wrap_eval_pool(env_pool: _FakePooledEnv) -> EvalPoolHandle:
    def _prepare_jobs(lane_jobs):
        args_list = []
        for lane_job in lane_jobs:
            if lane_job is None:
                raise AssertionError("lane_job should not be None after runner fill-in")
            args_list.append((int(lane_job.task_id), int(lane_job.episode_index)))
        env_pool.call_each("prepare_eval_job", args_list=args_list, kwargs_list=[])

    return EvalPoolHandle(env_pool=env_pool, prepare_jobs=_prepare_jobs)


def _wrap_eval_pool_allow_idle_lanes(env_pool: _FakePooledEnv) -> EvalPoolHandle:
    def _prepare_jobs(lane_jobs):
        active = [idx for idx, lane_job in enumerate(lane_jobs) if lane_job is not None]
        if not active:
            return
        pad_job = lane_jobs[active[0]]
        assert pad_job is not None
        prepared = [pad_job if lane_job is None else lane_job for lane_job in lane_jobs]
        args_list = [
            (int(lane_job.task_id), int(lane_job.episode_index))
            for lane_job in prepared
        ]
        env_pool.call_each("prepare_eval_job", args_list=args_list, kwargs_list=[])

    return EvalPoolHandle(env_pool=env_pool, prepare_jobs=_prepare_jobs)


def _wrap_lazy_init_eval_pool(num_envs: int) -> EvalPoolHandle:
    handle: EvalPoolHandle

    def _prepare_jobs(lane_jobs):
        _ = lane_jobs
        handle.env_pool = _FakePooledEnv(num_envs)

    handle = EvalPoolHandle(
        env_pool=None, num_envs=num_envs, prepare_jobs=_prepare_jobs
    )
    return handle


def _make_vec_env(success_at: int, task_desc: str) -> gym.vector.SyncVectorEnv:
    return gym.vector.SyncVectorEnv(
        [
            lambda: _TinyEnv(success_at=success_at, task_desc=task_desc),
            lambda: _TinyEnv(success_at=success_at, task_desc=task_desc),
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def _make_mixed_horizon_vec_env() -> gym.vector.SyncVectorEnv:
    return gym.vector.SyncVectorEnv(
        [
            lambda: _TinyEnv(
                success_at=999,
                max_steps=300,
                task_desc="short horizon task",
            ),
            lambda: _TinyEnv(
                success_at=400,
                max_steps=500,
                task_desc="long horizon task",
            ),
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def _make_async_vec_env(success_at: int, task_desc: str) -> gym.vector.AsyncVectorEnv:
    return gym.vector.AsyncVectorEnv(
        [
            partial(_TinyEnv, success_at=success_at, task_desc=task_desc),
            partial(_TinyEnv, success_at=success_at, task_desc=task_desc),
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def _make_async_vec_env_no_task(success_at: int) -> gym.vector.AsyncVectorEnv:
    return gym.vector.AsyncVectorEnv(
        [
            partial(_TinyEnvNoTask, success_at=success_at),
            partial(_TinyEnvNoTask, success_at=success_at),
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def _make_single_async_vec_env(
    success_at: int, task_desc: str
) -> gym.vector.AsyncVectorEnv:
    return gym.vector.AsyncVectorEnv(
        [
            partial(_TinyEnv, success_at=success_at, task_desc=task_desc),
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def _make_single_vec_env(success_at: int, task_desc: str) -> gym.vector.SyncVectorEnv:
    return gym.vector.SyncVectorEnv(
        [lambda: _TinyEnv(success_at=success_at, task_desc=task_desc)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def _make_single_nested_pixels_vec_env(
    success_at: int, task_desc: str
) -> gym.vector.SyncVectorEnv:
    return gym.vector.SyncVectorEnv(
        [lambda: _TinyNestedPixelsEnv(success_at=success_at, task_desc=task_desc)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


class TestSimEval:
    def test_rollout_compat_metrics_stop_at_first_terminal_step(self):
        rollout_data = {
            "done": torch.tensor(
                [
                    [False, True, False],
                    [False, False, True],
                ]
            ),
            "reward": torch.tensor(
                [
                    [-3.0, -2.0, 50.0],
                    [1.0, 2.0, 4.0],
                ]
            ),
            "success": torch.tensor(
                [
                    [False, False, True],
                    [False, False, True],
                ]
            ),
        }

        sum_rewards, max_rewards, successes, lengths, done_indices = (
            rollout_compat._terminal_rollout_metrics(rollout_data)
        )

        assert sum_rewards == pytest.approx([-5.0, 7.0])
        assert max_rewards == pytest.approx([-2.0, 4.0])
        assert successes == [False, True]
        assert lengths == [2, 3]
        assert done_indices.tolist() == [1, 2]

    def test_local_policy_adapter_select_action(self):
        policy = _TinyPolicy()
        adapter = LocalPolicyAdapter(policy=policy, device="cpu")
        batch = {"observation.state": torch.zeros((3, 2), dtype=torch.float32)}
        action = adapter.select_action(batch)
        assert action.shape == (3, 2)
        assert policy.last_kwargs == {}

    def test_local_policy_adapter_forwards_policy_kwargs(self):
        policy = _TinyPolicy()
        adapter = LocalPolicyAdapter(
            policy=policy,
            device="cpu",
            policy_kwargs={"decode_keep_k": 4, "decode_temperature": 0.7},
        )
        batch = {"observation.state": torch.zeros((2, 2), dtype=torch.float32)}
        action = adapter.select_action(batch)
        assert action.shape == (2, 2)
        assert policy.last_kwargs == {
            "decode_keep_k": 4,
            "decode_temperature": 0.7,
        }

    def test_local_policy_adapter_preserves_scalar_metadata(self):
        policy = _RecordingPolicy()
        adapter = LocalPolicyAdapter(policy=policy, device="cpu")
        batch = {
            "observation.state": torch.zeros((2, 2), dtype=torch.float32),
            "metadata.episode_index": [7, 8],
            "metadata.score": [np.float32(1.5), np.float32(2.5)],
            "metadata.done": [False, True],
            "task": ["left", "right"],
        }

        action = adapter.select_action(batch)

        assert action.shape == (2, 2)
        assert policy.observations is not None
        assert len(policy.observations) == 2
        np.testing.assert_array_equal(
            policy.observations[0]["observation.state"],
            np.zeros((2,), dtype=np.float32),
        )
        assert policy.observations[0]["metadata.episode_index"] == 7
        assert policy.observations[0]["metadata.score"] == 1.5
        assert policy.observations[0]["metadata.done"] is False
        assert policy.observations[0]["task"] == "left"
        assert policy.observations[1]["metadata.episode_index"] == 8
        assert policy.observations[1]["metadata.score"] == 2.5
        assert policy.observations[1]["metadata.done"] is True
        assert policy.observations[1]["task"] == "right"

    def test_local_policy_adapter_broadcasts_scalar_metadata_before_batch_keys(self):
        policy = _RecordingPolicy()
        adapter = LocalPolicyAdapter(policy=policy, device="cpu")
        batch = {
            "metadata.source": "rollout",
            "task": ["left", "right"],
            "observation.state": torch.zeros((2, 2), dtype=torch.float32),
        }

        action = adapter.select_action(batch)

        assert action.shape == (2, 2)
        assert policy.observations is not None
        assert [obs["metadata.source"] for obs in policy.observations] == [
            "rollout",
            "rollout",
        ]
        assert [obs["task"] for obs in policy.observations] == ["left", "right"]

    def test_local_policy_adapter_rejects_inconsistent_observation_batch_sizes(self):
        policy = _TinyPolicy()
        adapter = LocalPolicyAdapter(policy=policy, device="cpu")
        batch = {
            "task": ["left", "right"],
            "observation.state": torch.zeros((3, 2), dtype=torch.float32),
        }

        with pytest.raises(ValueError, match="Inconsistent observation batch sizes"):
            adapter.select_action(batch)

    def test_local_policy_adapter_rejects_zero_dimensional_observation_arrays(self):
        policy = _TinyPolicy()
        adapter = LocalPolicyAdapter(policy=policy, device="cpu")

        with pytest.raises(ValueError, match="zero-dimensional tensor"):
            adapter.select_action({"observation.state": torch.tensor(1.0)})
        with pytest.raises(ValueError, match="zero-dimensional numpy array"):
            adapter.select_action({"observation.state": np.array(1.0)})

    def test_local_policy_adapter_leaves_finite_action_bounds_to_rollout_sanitizer(
        self,
    ):
        policy = _OutOfBoundsActionPolicy()
        adapter = LocalPolicyAdapter(
            policy=policy,
            device="cpu",
            action_spec=ActionSpec(
                shape=(2,), dtype="float32", minimum=-1.0, maximum=1.0
            ),
        )

        action = adapter.select_action(
            {"observation.state": torch.zeros((1, 2), dtype=torch.float32)}
        )

        assert policy.action_spec is adapter.action_spec
        np.testing.assert_allclose(
            action.numpy(), np.array([[1.25, -1.25]], dtype=np.float32)
        )

    def test_local_policy_adapter_ignores_rollout_bookkeeping_fields(self):
        policy = _RecordingPolicy()
        adapter = LocalPolicyAdapter(policy=policy, device="cpu")
        batch = {
            "observation.state": torch.zeros((2, 2), dtype=torch.float32),
            "action": torch.ones((2, 2), dtype=torch.float32),
            "info": {"is_success": [False, True]},
        }

        action = adapter.select_action(batch)

        assert action.shape == (2, 2)
        assert policy.observations is not None
        assert "action" not in policy.observations[0]
        assert "action" not in policy.observations[1]
        assert "info" not in policy.observations[0]
        assert "info" not in policy.observations[1]

    def test_local_policy_adapter_collects_inference_diagnostics(self):
        policy = _DiagnosticTinyPolicy()
        adapter = LocalPolicyAdapter(policy=policy, device="cpu")
        batch = {"observation.state": torch.zeros((2, 2), dtype=torch.float32)}

        action = adapter.select_action(batch)
        summary = adapter.policy_diagnostics_summary()

        assert action.shape == (2, 2)
        assert summary["decode_keep_k"]["count"] == 2
        assert summary["decode_keep_k"]["mean"] == pytest.approx(6.0)
        assert summary["decode_keep_k"]["hist"] == {"4": 1, "8": 1}
        assert summary["decode_keep_k"]["frac"] == {"4": 0.5, "8": 0.5}
        assert summary["effective_decode_keep_k"]["hist"] == {"4": 1, "8": 1}
        assert summary["decode_steps"]["mean"] == pytest.approx(2.0)

    def test_policy_diagnostics_skip_raw_tokens_but_summarize_bools(self):
        acc = InferenceDiagnosticsAccumulator()

        acc.add(
            {
                "pred_action_token_ids": [[1, 2, 3], [4, 5]],
                "stopped_on_eos": [True, False],
            }
        )
        summary = acc.summary()

        assert "pred_action_token_ids" not in summary
        assert summary["stopped_on_eos"]["count"] == 2
        assert summary["stopped_on_eos"]["mean"] == pytest.approx(0.5)
        assert summary["stopped_on_eos"]["hist"] == {"0": 1, "1": 1}
        assert summary["stopped_on_eos"]["frac"] == {"0": 0.5, "1": 0.5}

    def test_evaluate_policy_on_env_pool_attaches_policy_diagnostics(self, monkeypatch):
        env_pool = _FakePooledEnv(num_envs=2)

        def _fake_eval(*, env, policy, **kwargs):
            _ = kwargs
            policy.select_action(
                {"observation.state": torch.zeros((2, 2), dtype=torch.float32)}
            )
            current_wave = env.prepared_waves[-1]
            return [
                TaskEvalResult(
                    task_group="",
                    task_id=0,
                    task_description=f"task-{task_id}",
                    sum_rewards=[1.0],
                    max_rewards=[1.0],
                    successes=[True],
                    lengths=[2],
                    video_paths=[],
                )
                for task_id, _episode_index in current_wave
            ]

        monkeypatch.setattr(sim_wave_retry, "evaluate_policy_on_pooled_env", _fake_eval)

        results = evaluate_policy_on_env_pool(
            tasks=[("suite", 0), ("suite", 1)],
            eval_pool=_wrap_eval_pool(env_pool),
            policy=_DiagnosticTinyPolicy(),
            num_eval_per_task=1,
            start_seed=0,
            device="cpu",
        )

        diagnostics = results["policy_diagnostics"]
        assert diagnostics["decode_keep_k"]["hist"] == {"4": 1, "8": 1}
        assert diagnostics["effective_decode_keep_k"]["mean"] == pytest.approx(6.0)

    def test_evaluate_policy_on_env_pool_single_task_uses_all_lanes(self, monkeypatch):
        env_pool = _FakePooledEnv(num_envs=2)

        def _fake_eval(
            *,
            env,
            max_episodes_rendered_by_env,
            videos_dirs_by_env,
            video_start_index_by_env,
            **kwargs,
        ):
            _ = (
                max_episodes_rendered_by_env,
                videos_dirs_by_env,
                video_start_index_by_env,
                kwargs,
            )
            current_wave = env.prepared_waves[-1]
            return [
                TaskEvalResult(
                    task_group="",
                    task_id=0,
                    task_description=f"task-{task_id}",
                    sum_rewards=[1.0],
                    max_rewards=[1.0],
                    successes=[True],
                    lengths=[2],
                    video_paths=[],
                )
                for task_id, _episode_index in current_wave
            ]

        monkeypatch.setattr(sim_wave_retry, "evaluate_policy_on_pooled_env", _fake_eval)

        results = evaluate_policy_on_env_pool(
            tasks=[("suite", 0)],
            eval_pool=_wrap_eval_pool(env_pool),
            policy=_TinyPolicy(),
            num_eval_per_task=5,
            start_seed=0,
            device="cpu",
        )

        assert results["overall"]["n_episodes"] == 5.0
        assert results["overall"]["success_rate"] == pytest.approx(1.0)
        assert results["per_task"]["suite/0"]["n_episodes"] == 5.0
        assert results["per_task"]["suite/0"]["success_rate"] == pytest.approx(1.0)
        assert len(env_pool.prepared_waves) == 3  # ceil(5 / 2)
        assert env_pool.prepared_waves[0] == [(0, 0), (0, 1)]
        assert env_pool.prepared_waves[1] == [(0, 2), (0, 3)]
        assert env_pool.prepared_waves[2] == [
            (0, 4),
            (0, 3),
        ]  # inactive lane holds last assignment

    def test_evaluate_policy_on_env_pool_non_divisible_tail(self, monkeypatch):
        env_pool = _FakePooledEnv(num_envs=4)

        def _fake_eval(
            *,
            env,
            max_episodes_rendered_by_env,
            videos_dirs_by_env,
            video_start_index_by_env,
            **kwargs,
        ):
            _ = (
                max_episodes_rendered_by_env,
                videos_dirs_by_env,
                video_start_index_by_env,
                kwargs,
            )
            current_wave = env.prepared_waves[-1]
            return [
                TaskEvalResult(
                    task_group="",
                    task_id=0,
                    task_description=f"task-{task_id}",
                    sum_rewards=[float(task_id)],
                    max_rewards=[float(task_id)],
                    successes=[True],
                    lengths=[3],
                    video_paths=[],
                )
                for task_id, _episode_index in current_wave
            ]

        monkeypatch.setattr(sim_wave_retry, "evaluate_policy_on_pooled_env", _fake_eval)

        results = evaluate_policy_on_env_pool(
            tasks=[("suite", 0), ("suite", 1)],
            eval_pool=_wrap_eval_pool(env_pool),
            policy=_TinyPolicy(),
            num_eval_per_task=3,
            start_seed=0,
            device="cpu",
        )

        assert results["overall"]["n_episodes"] == 6.0
        assert results["overall"]["success_rate"] == pytest.approx(1.0)
        assert results["per_task"]["suite/0"]["n_episodes"] == 3.0
        assert results["per_task"]["suite/0"]["success_rate"] == pytest.approx(1.0)
        assert results["per_task"]["suite/1"]["n_episodes"] == 3.0
        assert results["per_task"]["suite/1"]["success_rate"] == pytest.approx(1.0)
        assert len(env_pool.prepared_waves) == 2  # ceil(6 / 4)
        assert env_pool.prepared_waves[0] == [(0, 0), (1, 1), (0, 2), (1, 3)]
        assert env_pool.prepared_waves[1] == [(0, 4), (1, 5), (0, 2), (1, 3)]

    def test_evaluate_policy_on_env_pool_skips_permanently_idle_lanes(
        self, monkeypatch
    ):
        env_pool = _FakePooledEnv(num_envs=6)

        def _fake_eval(
            *,
            env,
            max_episodes_rendered_by_env,
            videos_dirs_by_env,
            video_start_index_by_env,
            **kwargs,
        ):
            _ = kwargs
            current_wave = env.prepared_waves[-1]
            out: list[TaskEvalResult] = []
            for lane_idx, (task_id, _episode_index) in enumerate(current_wave):
                assert int(max_episodes_rendered_by_env[lane_idx]) == 0
                assert videos_dirs_by_env[lane_idx] is None
                assert int(video_start_index_by_env[lane_idx]) == 0
                out.append(
                    TaskEvalResult(
                        task_group="",
                        task_id=0,
                        task_description=f"task-{task_id}",
                        sum_rewards=[1.0],
                        max_rewards=[1.0],
                        successes=[True],
                        lengths=[2],
                        video_paths=[],
                    )
                )
            return out

        monkeypatch.setattr(sim_wave_retry, "evaluate_policy_on_pooled_env", _fake_eval)

        results = evaluate_policy_on_env_pool(
            tasks=[("suite", i) for i in range(5)],
            eval_pool=_wrap_eval_pool_allow_idle_lanes(env_pool),
            policy=_TinyPolicy(),
            num_eval_per_task=1,
            start_seed=0,
            device="cpu",
        )

        assert results["overall"]["n_episodes"] == 5.0
        assert results["overall"]["success_rate"] == pytest.approx(1.0)

    def test_evaluate_policy_on_env_pool_video_cap_unique_indices(
        self, monkeypatch, tmp_path
    ):
        env_pool = _FakePooledEnv(num_envs=4)

        def _fake_eval(
            *,
            env,
            max_episodes_rendered_by_env,
            videos_dirs_by_env,
            video_start_index_by_env,
            **kwargs,
        ):
            _ = kwargs
            current_wave = env.prepared_waves[-1]
            out = []
            for lane_idx, (task_id, _episode_index) in enumerate(current_wave):
                video_paths: list[str] = []
                if (
                    int(max_episodes_rendered_by_env[lane_idx]) > 0
                    and videos_dirs_by_env[lane_idx] is not None
                ):
                    video_paths = [
                        str(
                            videos_dirs_by_env[lane_idx]
                            / f"eval_episode_{int(video_start_index_by_env[lane_idx])}.mp4"
                        )
                    ]
                out.append(
                    TaskEvalResult(
                        task_group="",
                        task_id=0,
                        task_description=f"task-{task_id}",
                        sum_rewards=[1.0],
                        max_rewards=[1.0],
                        successes=[True],
                        lengths=[2],
                        video_paths=video_paths,
                    )
                )
            return out

        monkeypatch.setattr(sim_wave_retry, "evaluate_policy_on_pooled_env", _fake_eval)

        results = evaluate_policy_on_env_pool(
            tasks=[("suite", 0)],
            eval_pool=_wrap_eval_pool(env_pool),
            policy=_TinyPolicy(),
            num_eval_per_task=4,
            start_seed=0,
            device="cpu",
            max_episodes_rendered_per_task=2,
            videos_dir=tmp_path / "videos",
        )

        assert results["overall"]["n_episodes"] == 4.0
        assert len(results["overall"]["video_paths"]) == 2
        assert len(set(results["overall"]["video_paths"])) == 2
        assert any(
            path.endswith("eval_episode_0.mp4")
            for path in results["overall"]["video_paths"]
        )
        assert any(
            path.endswith("eval_episode_1.mp4")
            for path in results["overall"]["video_paths"]
        )

    def test_evaluate_policy_on_env_pool_updates_global_progress_bar(self, monkeypatch):
        env_pool = _FakePooledEnv(num_envs=2)

        class _FakeTqdm:
            instances: list[_FakeTqdm] = []

            def __init__(self, *args, total=None, **kwargs):
                self.total = total
                self.updated: list[int] = []
                self.postfixes: list[dict[str, str]] = []
                self.messages: list[str] = []
                self.events: list[tuple[str, object]] = []
                _FakeTqdm.instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, n=1):
                self.updated.append(int(n))
                self.events.append(("update", int(n)))

            def set_postfix(self, refresh=True, **kwargs):
                _ = refresh
                self.postfixes.append(dict(kwargs))
                self.events.append(("postfix", dict(kwargs)))

            def write(self, message):
                self.messages.append(str(message))

        def _fake_eval(
            *,
            env,
            max_episodes_rendered_by_env,
            videos_dirs_by_env,
            video_start_index_by_env,
            **kwargs,
        ):
            _ = (
                max_episodes_rendered_by_env,
                videos_dirs_by_env,
                video_start_index_by_env,
                kwargs,
            )
            current_wave = env.prepared_waves[-1]
            return [
                TaskEvalResult(
                    task_group="",
                    task_id=0,
                    task_description=f"task-{task_id}",
                    sum_rewards=[1.0],
                    max_rewards=[1.0],
                    successes=[True],
                    lengths=[2],
                    video_paths=[],
                )
                for task_id, _episode_index in current_wave
            ]

        monkeypatch.setattr(sim_wave_retry, "evaluate_policy_on_pooled_env", _fake_eval)
        monkeypatch.setattr(sim_runner, "tqdm", _FakeTqdm)

        results = evaluate_policy_on_env_pool(
            tasks=[("suite", 0)],
            eval_pool=_wrap_eval_pool(env_pool),
            policy=_TinyPolicy(),
            num_eval_per_task=5,
            start_seed=0,
            device="cpu",
        )

        assert results["overall"]["n_episodes"] == 5.0
        assert len(_FakeTqdm.instances) == 1
        bar = _FakeTqdm.instances[0]
        assert bar.total == 5
        assert sum(bar.updated) == 5
        assert bar.updated == [2, 2, 1]
        assert bar.postfixes
        assert bar.postfixes[0]["succ_rate"] == "0.0%"
        assert bar.postfixes[-1]["succ_rate"] == "100.0%"
        assert bar.events[1][0] == "postfix"
        assert bar.events[1][1]["succ_rate"] == "100.0%"
        assert bar.events[2] == ("update", 2)
        assert bar.messages == []

    def test_evaluate_policy_on_env_pool_uses_initialized_pool_after_prepare(
        self, monkeypatch
    ):
        captured_envs: list[object] = []

        def _fake_eval(*, env, **kwargs):
            _ = kwargs
            captured_envs.append(env)
            return [
                TaskEvalResult(
                    task_group="suite",
                    task_id=0,
                    task_description="task 0",
                    sum_rewards=[1.0],
                    max_rewards=[1.0],
                    successes=[True],
                    lengths=[1],
                    video_paths=[],
                ),
                TaskEvalResult(
                    task_group="suite",
                    task_id=1,
                    task_description="task 1",
                    sum_rewards=[1.0],
                    max_rewards=[1.0],
                    successes=[True],
                    lengths=[1],
                    video_paths=[],
                ),
            ]

        monkeypatch.setattr(sim_wave_retry, "evaluate_policy_on_pooled_env", _fake_eval)

        result = evaluate_policy_on_env_pool(
            tasks=[("suite", 0), ("suite", 1)],
            eval_pool=_wrap_lazy_init_eval_pool(num_envs=2),
            policy=_TinyPolicy(),
            num_eval_per_task=1,
        )

        assert len(captured_envs) == 1
        assert isinstance(captured_envs[0], _FakePooledEnv)
        assert result["overall"]["n_episodes"] == 2.0

    def test_evaluate_policy_on_env_pool_rejects_pool_lane_count_mismatch(self):
        handle: EvalPoolHandle

        def _prepare_jobs(lane_jobs):
            _ = lane_jobs
            handle.env_pool = _FakePooledEnv(num_envs=1)

        handle = EvalPoolHandle(
            env_pool=None,
            num_envs=2,
            prepare_jobs=_prepare_jobs,
        )

        with pytest.raises(RuntimeError, match="lane count mismatch"):
            evaluate_policy_on_env_pool(
                tasks=[("suite", 0), ("suite", 1)],
                eval_pool=handle,
                policy=_TinyPolicy(),
                num_eval_per_task=1,
            )

    def test_evaluate_policy_on_env_pool_rejects_rollout_result_count_mismatch(
        self, monkeypatch
    ):
        env_pool = _FakePooledEnv(num_envs=2)

        def _fake_eval(*, env, **kwargs):
            _ = (env, kwargs)
            return [
                TaskEvalResult(
                    task_group="suite",
                    task_id=0,
                    task_description="task 0",
                    sum_rewards=[1.0],
                    max_rewards=[1.0],
                    successes=[True],
                    lengths=[1],
                    video_paths=[],
                )
            ]

        monkeypatch.setattr(sim_wave_retry, "evaluate_policy_on_pooled_env", _fake_eval)

        with pytest.raises(RuntimeError, match="expected 2, got 1"):
            evaluate_policy_on_env_pool(
                tasks=[("suite", 0), ("suite", 1)],
                eval_pool=_wrap_eval_pool(env_pool),
                policy=_TinyPolicy(),
                num_eval_per_task=1,
            )

    def test_evaluate_policy_on_env_pool_retries_retryable_rollout_failure(
        self, monkeypatch
    ):
        pools: list[_FakePooledEnv] = []
        handle: EvalPoolHandle

        def _prepare_jobs(lane_jobs):
            pool = _FakePooledEnv(num_envs=2)
            pools.append(pool)
            handle.env_pool = pool
            args_list = []
            for lane_job in lane_jobs:
                if lane_job is None:
                    raise AssertionError(
                        "lane_job should not be None after runner fill-in"
                    )
                args_list.append((int(lane_job.task_id), int(lane_job.episode_index)))
            pool.call_each("prepare_eval_job", args_list=args_list, kwargs_list=[])

        handle = EvalPoolHandle(env_pool=None, num_envs=2, prepare_jobs=_prepare_jobs)
        calls = 0

        def _fake_eval(*, env, **kwargs):
            _ = kwargs
            nonlocal calls
            calls += 1
            if calls == 1:
                raise EOFError("native worker pipe closed")
            current_wave = env.prepared_waves[-1]
            return [
                TaskEvalResult(
                    task_group="",
                    task_id=0,
                    task_description=f"task-{task_id}",
                    sum_rewards=[1.0],
                    max_rewards=[1.0],
                    successes=[True],
                    lengths=[2],
                    video_paths=[],
                )
                for task_id, _episode_index in current_wave
            ]

        monkeypatch.setattr(sim_wave_retry, "evaluate_policy_on_pooled_env", _fake_eval)

        results = evaluate_policy_on_env_pool(
            tasks=[("suite", 0), ("suite", 1)],
            eval_pool=handle,
            policy=_TinyPolicy(),
            num_eval_per_task=1,
            rollout_failure_retries=1,
        )

        assert calls == 2
        assert len(pools) == 2
        assert pools[0].closed is True
        assert pools[0].prepared_waves == [[(0, 0), (1, 1)]]
        assert pools[1].prepared_waves == [[(0, 0), (1, 1)]]
        assert results["overall"]["n_episodes"] == 2.0
        assert results["overall"]["success_rate"] == pytest.approx(1.0)

    def test_rollout_compat_times_out_async_step(self, monkeypatch):
        env = _make_single_async_vec_env(success_at=2, task_desc="tiny")
        policy = LocalPolicyAdapter(policy=_TinyPolicy(), device="cpu")

        def _timeout_step_wait(*args, **kwargs):
            _ = (args, kwargs)
            raise multiprocessing.TimeoutError("step timed out")

        monkeypatch.setattr(env, "step_wait", _timeout_step_wait)
        try:
            with pytest.raises(
                RuntimeError, match="Timed out waiting for async env step"
            ):
                rollout_compat.evaluate_policy_on_pooled_env(
                    env=env,
                    policy=policy,
                    step_timeout_sec=0.01,
                )
        finally:
            env.close(terminate=True)

    def test_rollout_compat_times_out_async_reset(self, monkeypatch):
        env = _make_single_async_vec_env(success_at=2, task_desc="tiny")
        policy = LocalPolicyAdapter(policy=_TinyPolicy(), device="cpu")

        def _timeout_reset_wait(*args, **kwargs):
            _ = (args, kwargs)
            raise multiprocessing.TimeoutError("reset timed out")

        monkeypatch.setattr(env, "reset_wait", _timeout_reset_wait)
        try:
            with pytest.raises(
                RuntimeError, match="Timed out waiting for async env reset"
            ):
                rollout_compat.evaluate_policy_on_pooled_env(
                    env=env,
                    policy=policy,
                    step_timeout_sec=0.01,
                )
        finally:
            env.close(terminate=True)

    def test_rollout_compat_uses_longest_lane_horizon(self):
        env = _make_mixed_horizon_vec_env()
        policy = LocalPolicyAdapter(policy=_TinyPolicy(), device="cpu")

        try:
            results = rollout_compat.evaluate_policy_on_pooled_env(
                env=env,
                policy=policy,
            )
        finally:
            env.close()

        assert [result.task_description for result in results] == [
            "short horizon task",
            "long horizon task",
        ]
        assert [result.lengths for result in results] == [[300], [400]]
        assert [result.successes for result in results] == [[False], [True]]

    def test_rollout_compat_rejects_non_finite_actions_before_step(self):
        env = gym.vector.SyncVectorEnv(
            [lambda: _RecordingTinyEnv(success_at=1, task_desc="tiny")],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        policy = LocalPolicyAdapter(policy=_BadActionPolicy(), device="cpu")

        try:
            with pytest.raises(ValueError, match="non-finite"):
                rollout_compat.evaluate_policy_on_pooled_env(env=env, policy=policy)
            actions = list(env.envs[0].actions)
        finally:
            env.close()

        assert actions == []

    def test_rollout_compat_clips_out_of_bounds_actions_before_step(self):
        env = gym.vector.SyncVectorEnv(
            [lambda: _RecordingTinyEnv(success_at=1, task_desc="tiny")],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        policy = LocalPolicyAdapter(policy=_OutOfBoundsActionPolicy(), device="cpu")

        try:
            rollout_compat.evaluate_policy_on_pooled_env(env=env, policy=policy)
            actions = list(env.envs[0].actions)
        finally:
            env.close()

        assert len(actions) == 1
        np.testing.assert_allclose(actions[0], np.array([1.0, -1.0], dtype=np.float32))
