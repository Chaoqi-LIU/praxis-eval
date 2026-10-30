# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Gymnasium vector env over the evaluation env pools, for RL rollouts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space, create_empty_array
from gymnasium.vector.vector_env import AutoresetMode

from praxis_eval.contracts import ActionSpec
from praxis_eval.envs.eval_pool import EvalLaneJob, EvalPoolHandle
from praxis_eval.envs.factory import build_env_config, make_eval_async_pool
from praxis_eval.evaluation.config import require_positive_int
from praxis_eval.evaluation.sim.adapters import (
    _action_spec_without_bounds,
    policy_observation_batch,
    split_policy_observations,
)
from praxis_eval.evaluation.sim.rollout_compat import (
    _RolloutEnvCompat,
    postprocess_action,
    process_observation,
    step_successes,
)
from praxis_eval.policies.actions import normalize_batched_action
from praxis_eval.processing import make_env_pre_post_processors
from praxis_eval.registry import get_driver


def make_vector_env(
    env: str,
    *,
    lanes: Sequence[tuple[str, int]],
    env_kwargs: Mapping[str, Any] | None = None,
    episode_length: int | None = None,
    seed: int = 0,
    debug_verbose: bool = False,
    mp_context: str | None = None,
) -> EvalVectorEnv:
    """Build an ``EvalVectorEnv`` from the same env pool ``evaluate(env, ...)`` uses.

    Args:
        env: Registered driver name with an eval pool, such as ``"libero"``.
        lanes: One ``(task_group, task_id)`` per lane, as returned by
            ``praxis_eval.envs.factory.list_tasks``. Lanes may repeat or mix tasks.
        env_kwargs: Benchmark config, the same as ``EvalConfig.env_kwargs``.
        episode_length: Steps before a lane is truncated. ``None`` uses each
            task's own horizon, the one ``evaluate()`` uses.
        seed: Base seed. The k-th episode of a task resets with the seed the eval
            runner would give it, ``seed + episode_index``.
        debug_verbose: Keep benchmark startup output.
        mp_context: Multiprocessing start method of the env workers, the same
            as ``EvalConfig.mp_context``.
    """
    contract = get_driver(env).contract
    env_cfg = {**dict(env_kwargs or {}), "type": contract.env_type}
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=build_env_config(env_cfg),
        policy_cfg=None,
    )
    lanes = [(str(task_group), int(task_id)) for task_group, task_id in lanes]
    eval_pool = make_eval_async_pool(
        env_cfg,
        tasks=list(dict.fromkeys(lanes)),
        n_envs=len(lanes),
        debug_verbose=debug_verbose,
        mp_context=mp_context,
    )
    return EvalVectorEnv(
        eval_pool,
        lanes=lanes,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        action_spec=contract.action,
        episode_length=episode_length,
        seed=seed,
    )


class EvalVectorEnv(gym.vector.VectorEnv):
    """Gymnasium vector env over one eval pool, one task per lane.

    Observations are the batched values ``evaluate()`` passes to policies: numpy
    arrays with a leading lane axis, ``task`` as a list of strings, and scalar
    metadata shared by all lanes. ``observation_space`` describes the arrays.
    Actions are float32 ``(num_envs, action_dim)`` and go through the same
    validation and clipping as eval actions.

    A lane is done when its env terminates or after ``episode_length`` steps,
    which is reported as truncation. Done lanes start their next episode inside
    the same ``step()`` call, and ``info`` has the layout of gymnasium's
    ``SyncVectorEnv`` in SAME_STEP mode: ``info["final_obs"]`` holds each done
    lane's last observation, ``info["final_info"]`` its last step info, whose
    ``is_success`` is the success ``evaluate()`` records, and a done lane's
    top-level info is its reset info.

    Episodes follow the eval runner's numbering: the k-th episode of a task runs
    the job ``EvalLaneJob(task_group, task_id, k, k * n_tasks + task_index)`` and
    resets with seed ``seed + episode_index``, so benchmarks pick init states and
    scenes the same way ``evaluate()`` does.
    """

    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(
        self,
        eval_pool: EvalPoolHandle,
        *,
        lanes: Sequence[tuple[str, int]],
        env_preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
        env_postprocessor: Callable[[dict[str, Any]], dict[str, Any]],
        action_spec: ActionSpec | None,
        episode_length: int | None = None,
        seed: int = 0,
    ) -> None:
        self.lanes = tuple((str(group), int(task_id)) for group, task_id in lanes)
        self.num_envs = len(self.lanes)
        if self.num_envs < 1 or eval_pool.num_envs != self.num_envs:
            raise ValueError(
                f"Expected one eval pool lane per task lane, got {self.num_envs} "
                f"lanes and {eval_pool.num_envs} pool lanes."
            )
        self._eval_pool = eval_pool
        self._env_preprocessor = env_preprocessor
        self._env_postprocessor = env_postprocessor
        # Bounds are not errors: out-of-range actions are clipped as in eval.
        self._action_spec = _action_spec_without_bounds(action_spec)
        self._seed = int(seed)
        self._task_index = {
            task: idx for idx, task in enumerate(dict.fromkeys(self.lanes))
        }
        self._episode_counts = dict.fromkeys(self._task_index, 0)
        self._elapsed: np.ndarray | None = None

        # Build the workers with the jobs the first reset() starts.
        self._lane_jobs = [self._next_job(task) for task in self.lanes]
        self._episode_counts = dict.fromkeys(self._task_index, 0)
        try:
            eval_pool.prepare_jobs(list(self._lane_jobs))
            env_pool = eval_pool.env_pool
            if env_pool is None:
                raise RuntimeError("Eval pool prepare_jobs did not build env_pool.")
            _check_supports_masked_reset(env_pool)
            self._env = _RolloutEnvCompat(env_pool)
            self.single_action_space = env_pool.single_action_space
            self.action_space = env_pool.action_space
            # Every raw observation fits the pool's space (the async pool's shared
            # memory is laid out from it), so processing a zero observation gives
            # the processed keys, shapes, and dtypes.
            self.single_observation_space = _array_observation_space(
                self._process(
                    create_empty_array(
                        env_pool.single_observation_space, self.num_envs, np.zeros
                    )
                )
            )
            self.observation_space = batch_space(
                self.single_observation_space, self.num_envs
            )
            if episode_length is None:
                horizons = env_pool.call("_max_episode_steps")
            else:
                limit = require_positive_int(episode_length, name="episode_length")
                horizons = [limit] * self.num_envs
            self._episode_lengths = np.asarray(horizons, dtype=np.int64)
        except BaseException:
            eval_pool.close(terminate=True)
            raise

    @property
    def lane_jobs(self) -> tuple[EvalLaneJob, ...]:
        """The eval job each lane's current episode runs."""
        return tuple(self._lane_jobs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode in every lane.

        Passing ``seed`` replaces the base seed and restarts the per-task episode
        counters, so the episodes repeat those of a new env built with ``seed``.
        """
        if options is not None:
            raise ValueError("EvalVectorEnv.reset() does not take options.")
        if seed is not None:
            self._seed = int(seed)
            self._episode_counts = dict.fromkeys(self._task_index, 0)
        self._elapsed = np.zeros(self.num_envs, dtype=np.int64)
        return self._start_episodes(np.ones(self.num_envs, dtype=bool))

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if self._elapsed is None:
            raise RuntimeError("Call reset() before step().")
        actions = normalize_batched_action(
            actions, batch_size=self.num_envs, action_spec=self._action_spec
        )
        actions = postprocess_action(torch.from_numpy(actions), self._env_postprocessor)
        raw_observation, rewards, terminations, truncations, step_info = self._env.step(
            actions
        )
        self._elapsed += 1
        terminations = np.asarray(terminations, dtype=bool)
        truncations = np.asarray(truncations, dtype=bool) | (
            self._elapsed >= self._episode_lengths
        )
        done = terminations | truncations
        if not done.any():
            observation = self._process(raw_observation)
            return observation, rewards, terminations, truncations, step_info

        successes = step_successes(step_info, self.num_envs)
        final_observations = split_policy_observations(self._process(raw_observation))
        observation, reset_info = self._start_episodes(done)
        # Built lane by lane like gymnasium's SyncVectorEnv in SAME_STEP mode: a
        # done lane's step info moves to final_info and its reset info replaces it.
        info: dict[str, Any] = {}
        for lane in range(self.num_envs):
            if not done[lane]:
                info = self._add_info(info, _lane_info(step_info, lane), lane)
                continue
            final_info = {
                **_lane_info(step_info, lane),
                "is_success": bool(successes[lane]),
            }
            info = self._add_info(
                info,
                {"final_obs": final_observations[lane], "final_info": final_info},
                lane,
            )
            info = self._add_info(info, _lane_info(reset_info, lane), lane)
        return observation, rewards, terminations, truncations, info

    def close_extras(self, **kwargs: Any) -> None:
        self._eval_pool.close(**kwargs)

    def _next_job(self, task: tuple[str, int]) -> EvalLaneJob:
        # Same numbering as the eval runner: eval_idx is the per-task count.
        eval_idx = self._episode_counts[task]
        self._episode_counts[task] += 1
        episode_index = eval_idx * len(self._task_index) + self._task_index[task]
        return EvalLaneJob(task[0], task[1], eval_idx, episode_index)

    def _start_episodes(
        self, mask: np.ndarray
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        for lane in np.flatnonzero(mask):
            self._lane_jobs[lane] = self._next_job(self.lanes[lane])
        # Lanes outside the mask get their current job again, which only sets
        # what their next reset will use.
        self._eval_pool.prepare_jobs(list(self._lane_jobs))
        env_pool = self._eval_pool.env_pool
        assert env_pool is not None
        raw_observation, info = env_pool.reset(
            seed=[self._seed + job.episode_index for job in self._lane_jobs],
            options={"reset_mask": mask},
        )
        assert self._elapsed is not None
        self._elapsed[mask] = 0
        return self._process(raw_observation), info

    def _process(self, raw_observation: dict[str, Any]) -> dict[str, Any]:
        return policy_observation_batch(
            process_observation(self._env, raw_observation, self._env_preprocessor)
        )


def _array_observation_space(observation: dict[str, Any]) -> gym.spaces.Dict:
    """Return the single-lane space of a batched observation's array values.

    Bounds are the widest the dtype allows, since processing does not map raw
    space bounds to processed ones.
    """
    spaces: dict[str, gym.spaces.Space] = {}
    for key, value in observation.items():
        if not isinstance(value, np.ndarray):
            continue
        if np.issubdtype(value.dtype, np.floating):
            low, high = -np.inf, np.inf
        elif np.issubdtype(value.dtype, np.integer):
            low, high = np.iinfo(value.dtype).min, np.iinfo(value.dtype).max
        else:
            raise TypeError(
                f"Observation key {key!r} has dtype {value.dtype}, which no Box "
                "space describes."
            )
        spaces[key] = gym.spaces.Box(low, high, value.shape[1:], value.dtype)
    return gym.spaces.Dict(spaces)


def _lane_info(info: dict[str, Any], lane: int) -> dict[str, Any]:
    """Return one lane's info from a batched info dict (inverts ``_add_info``)."""
    lane_info: dict[str, Any] = {}
    for key, value in info.items():
        if key.startswith("_") and key[1:] in info:
            continue  # a mask
        if not info[f"_{key}"][lane]:
            continue
        if isinstance(value, dict):
            lane_info[key] = _lane_info(value, lane)
        elif value.dtype == np.bool_:
            # A Python bool, so _add_info batches it into a bool array again.
            lane_info[key] = bool(value[lane])
        else:
            lane_info[key] = value[lane]
    return lane_info


def _check_supports_masked_reset(env_pool: gym.vector.VectorEnv) -> None:
    """Fail fast if a masked reset would not cleanly restart finished lanes.

    Finished lanes are reset explicitly, right after the step that ended them. A
    SAME_STEP pool has already reset them, and gymnasium's async worker clears
    its pending NEXT_STEP autoreset on an explicit reset only when it uses shared
    memory. Otherwise a lane would be reset twice and drop the next action.
    """
    same_step = getattr(env_pool, "autoreset_mode", None) == AutoresetMode.SAME_STEP
    no_shared_memory = isinstance(env_pool, gym.vector.AsyncVectorEnv) and not bool(
        env_pool.shared_memory
    )
    if same_step or no_shared_memory:
        raise ValueError(
            "EvalVectorEnv needs a NEXT_STEP or DISABLED autoreset pool, and "
            "shared_memory=True for async pools."
        )
