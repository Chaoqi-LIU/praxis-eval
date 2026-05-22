# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Custom AsyncVectorEnv with optional ``dummy_env_fn`` bootstrap."""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import Space
from gymnasium.core import Env
from gymnasium.error import AlreadyPendingCallError, CustomSpaceError, NoAsyncCallError
from gymnasium.vector.async_vector_env import AsyncState, _async_worker
from gymnasium.vector.utils import (
    CloudpickleWrapper,
    batch_space,
    clear_mpi_env_vars,
    create_empty_array,
    create_shared_memory,
    read_from_shared_memory,
)
from gymnasium.vector.vector_env import AutoresetMode


class AsyncVectorEnvCallTimeout(RuntimeError):
    """Raised when ``call_each`` times out waiting on one or more lanes."""

    def __init__(
        self, *, method: str, timeout_sec: float | int, pending_lanes: list[int]
    ):
        self.details = {"method": str(method), "pending_lanes": list(pending_lanes)}
        super().__init__(
            "AsyncVectorEnv.call_each timed out after "
            f"{float(timeout_sec):.3f}s for method={method!r}, "
            f"pending_lanes={pending_lanes}."
        )


class AsyncVectorEnv(gym.vector.AsyncVectorEnv):
    """Drop-in replacement for ``gymnasium.vector.AsyncVectorEnv`` that
    accepts an optional *dummy_env_fn* used only to read
    observation/action spaces in the parent process.

    When ``dummy_env_fn`` is provided the parent never calls
    ``env_fns[0]()`` — real environment constructors run exclusively
    inside child workers, avoiding EGL/MuJoCo state corruption under
    fork-based multiprocessing.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        *,
        dummy_env_fn: Callable[[], Env] | None = None,
        shared_memory: bool = True,
        copy: bool = True,
        context: str | None = None,
        daemon: bool = True,
        worker: Callable | None = None,
        observation_mode: str | Space = "same",
        autoreset_mode: str | AutoresetMode = AutoresetMode.NEXT_STEP,
    ):
        # ---- store attributes (mirrors gymnasium) ----
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        self.context = context
        self.daemon = daemon
        self.worker = worker
        self.observation_mode = observation_mode
        self.autoreset_mode = (
            autoreset_mode
            if isinstance(autoreset_mode, AutoresetMode)
            else AutoresetMode(autoreset_mode)
        )

        self.num_envs = len(env_fns)

        # >>> Only change from gymnasium: use dummy_env_fn when provided <<<
        dummy_env = (dummy_env_fn or env_fns[0])()

        self.metadata = dummy_env.metadata
        self.metadata["autoreset_mode"] = self.autoreset_mode
        self.render_mode = dummy_env.render_mode

        self.single_action_space = dummy_env.action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        if isinstance(observation_mode, tuple) and len(observation_mode) == 2:
            assert isinstance(observation_mode[0], Space)
            assert isinstance(observation_mode[1], Space)
            self.observation_space, self.single_observation_space = observation_mode
        else:
            if observation_mode == "same":
                self.single_observation_space = dummy_env.observation_space
                self.observation_space = batch_space(
                    self.single_observation_space, self.num_envs
                )
            elif observation_mode == "different":
                if dummy_env_fn is not None:
                    raise ValueError(
                        "AsyncVectorEnv cannot infer observation_mode='different' "
                        "from real env_fns when dummy_env_fn is set. Pass explicit "
                        "(batch_space, single_space) observation spaces instead."
                    )
                env_spaces = [env().observation_space for env in self.env_fns]
                self.single_observation_space = env_spaces[0]
                from gymnasium.vector.utils import batch_differing_spaces

                self.observation_space = batch_differing_spaces(env_spaces)
            else:
                raise ValueError(
                    f"Invalid `observation_mode`, expected: 'same' or 'different' "
                    f"or tuple of single and batch observation space, actual got {observation_mode}"
                )

        dummy_env.close()
        del dummy_env

        # ---- shared memory / observation buffer (identical to gymnasium) ----
        ctx = cast(Any, multiprocessing.get_context(context))
        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(
                    self.single_observation_space, n=self.num_envs, ctx=ctx
                )
                self.observations = read_from_shared_memory(
                    self.single_observation_space, _obs_buffer, n=self.num_envs
                )
            except CustomSpaceError as e:
                raise ValueError(
                    "Using `AsyncVector(..., shared_memory=True)` caused an error, "
                    "you can disable this feature with `shared_memory=False` however this is slower."
                ) from e
        else:
            _obs_buffer = None
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )

        # ---- spawn worker processes (identical to gymnasium) ----
        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = worker or _async_worker
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                        self.autoreset_mode,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_spaces()

    def _abort_on_call_timeout(self) -> None:
        """Destroy the pool on ``call_each`` timeout.

        Re-using an AsyncVectorEnv after a timed-out ``call_each`` is unsafe:
        worker processes may still be executing the first call and their
        pending responses would shift the pipe read offsets of any subsequent
        command, surfacing as spurious AttributeError / garbled return values
        long after the original timeout. Closing parent pipes + terminating
        workers makes every subsequent operation raise immediately.
        """
        with suppress(Exception):
            self.close(terminate=True)
        self._state = AsyncState.DEFAULT

    def _poll_parent_pipes(self, timeout: int | float | None = None) -> bool:
        """Poll all parent pipes until timeout, returning ``True`` if all are ready."""
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + float(timeout)
        for pipe in self.parent_pipes:
            remaining = max(end_time - time.perf_counter(), 0.0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(remaining)):
                return False
        return True

    def call_each(
        self,
        name: str,
        *,
        args_list: Sequence[Sequence[Any] | Any] | None = None,
        kwargs_list: Sequence[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> tuple[Any, ...]:
        """Call ``name`` on each worker with lane-specific args/kwargs.

        This is the per-lane companion to ``call``/``call_async`` where each
        worker receives its own argument tuple.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `call_each` while waiting for `{self._state.value}` to complete.",
                str(self._state.value),
            )

        n_envs = len(self.parent_pipes)
        if args_list is None:
            args_list = [()] * n_envs
        if kwargs_list is None:
            kwargs_list = [{}] * n_envs

        if len(args_list) != n_envs:
            raise ValueError(
                f"args_list length must be {n_envs}, got {len(args_list)}."
            )
        if len(kwargs_list) != n_envs:
            raise ValueError(
                f"kwargs_list length must be {n_envs}, got {len(kwargs_list)}."
            )

        normalized_args: list[tuple[Any, ...]] = []
        for lane_args in args_list:
            if isinstance(lane_args, tuple):
                normalized_args.append(lane_args)
            elif isinstance(lane_args, list):
                normalized_args.append(tuple(lane_args))
            else:
                normalized_args.append((lane_args,))

        for idx, pipe in enumerate(self.parent_pipes):
            lane_kwargs = dict(kwargs_list[idx])
            pipe.send(("_call", (name, normalized_args[idx], lane_kwargs)))
        self._state = AsyncState.WAITING_CALL
        timeout_sec = 0.0 if timeout is None else float(timeout)

        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_each` receive path without pending call.",
                AsyncState.WAITING_CALL.value,
            )

        poll_fn = getattr(self, "_poll_pipe_envs", None)
        ready = (
            bool(poll_fn(timeout))
            if callable(poll_fn)
            else self._poll_parent_pipes(timeout)
        )
        if not ready:
            pending_lanes = [
                lane_idx
                for lane_idx, pipe in enumerate(self.parent_pipes)
                if not pipe.poll(0.0)
            ]
            if not pending_lanes:
                pending_lanes = list(range(len(self.parent_pipes)))
            self._abort_on_call_timeout()
            raise AsyncVectorEnvCallTimeout(
                method=name,
                timeout_sec=timeout_sec,
                pending_lanes=pending_lanes,
            )

        deadline = time.perf_counter() + timeout_sec if timeout is not None else None
        results_with_status: list[tuple[Any, bool]] = []
        for lane_idx, pipe in enumerate(self.parent_pipes):
            if deadline is not None:
                remaining = max(deadline - time.perf_counter(), 0.0)
                if not pipe.poll(remaining):
                    pending_lanes = [lane_idx]
                    pending_lanes.extend(range(lane_idx + 1, len(self.parent_pipes)))
                    self._abort_on_call_timeout()
                    raise AsyncVectorEnvCallTimeout(
                        method=name,
                        timeout_sec=timeout_sec,
                        pending_lanes=pending_lanes,
                    )
            results_with_status.append(pipe.recv())

        results, successes = zip(*results_with_status, strict=True)
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        return results
