"""Tests for the custom AsyncVectorEnv with real worker processes."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from praxis_eval.envs.async_vector_env import AsyncVectorEnv


class _CountingEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self._steps += 1
        return np.full(1, self._steps, np.float32), 0.0, False, False, {}


def test_async_vector_env_steps_real_workers_with_installed_gymnasium():
    envs = AsyncVectorEnv([_CountingEnv, _CountingEnv], dummy_env_fn=_CountingEnv)
    try:
        obs, _ = envs.reset(seed=0)
        np.testing.assert_array_equal(obs, [[0.0], [0.0]])
        obs, *_ = envs.step(np.zeros((2, 1), np.float32))
        np.testing.assert_array_equal(obs, [[1.0], [1.0]])
    finally:
        envs.close()
