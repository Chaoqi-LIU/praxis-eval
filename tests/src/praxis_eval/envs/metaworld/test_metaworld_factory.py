"""Unit tests for MetaWorld factory registration and async eval helpers."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from gymnasium import spaces

from praxis_eval.envs.eval_pool import EvalLaneJob
from praxis_eval.envs.factory import (
    available_env_types,
    available_eval_pool_env_types,
    build_env_config,
    infer_eval_env_target,
    list_tasks,
    make_env,
)
from praxis_eval.envs.metaworld.config import MetaworldEnvConfig
from praxis_eval.envs.metaworld.tasks import (
    MT50_GROUPS,
    expand_task_selectors,
    get_task_description,
    infer_metaworld_eval_target_from_dataset,
    list_metaworld_tasks,
    resolve_task_name,
)


class TestMetaworldTasks:
    def test_mt50_expands_to_all_difficulty_groups(self) -> None:
        assert expand_task_selectors("mt50") == list(MT50_GROUPS)
        assert expand_task_selectors("easy,medium,hard,very_hard") == list(MT50_GROUPS)

    def test_list_mt50_tasks_keeps_difficulty_groups(self) -> None:
        cfg = MetaworldEnvConfig(task="mt50")
        tasks = list_metaworld_tasks({}, cfg)
        assert len(tasks) == 50
        assert sum(1 for group, _task_id in tasks if group == "easy") == 28
        assert sum(1 for group, _task_id in tasks if group == "medium") == 11
        assert sum(1 for group, _task_id in tasks if group == "hard") == 6
        assert sum(1 for group, _task_id in tasks if group == "very_hard") == 5

    def test_resolve_task_name(self) -> None:
        assert resolve_task_name("easy", 0) == "button-press-v3"
        assert resolve_task_name("very_hard", 4) == "pick-place-wall-v3"
        assert resolve_task_name("metaworld-reach-v3", 0) == "reach-v3"
        with pytest.raises(ValueError, match="out of range"):
            resolve_task_name("very_hard", 5)

    def test_task_description(self) -> None:
        assert get_task_description("reach-v3") == "Reach a goal position"

    def test_infer_eval_target(self) -> None:
        assert infer_metaworld_eval_target_from_dataset("metaworld_mt50") == (
            "metaworld",
            "mt50",
        )
        assert infer_metaworld_eval_target_from_dataset("metaworld_easy") == (
            "metaworld",
            "easy",
        )
        assert infer_metaworld_eval_target_from_dataset("metaworld_reach-v3") == (
            "metaworld",
            "reach-v3",
        )
        assert infer_metaworld_eval_target_from_dataset("libero_10") is None
        assert infer_eval_env_target("metaworld_mt50") == ("metaworld", "mt50")


class TestMetaworldFactoryRegistration:
    def test_available_env_types(self) -> None:
        assert "metaworld" in available_env_types()
        assert "metaworld" in available_eval_pool_env_types()

    def test_build_env_config(self) -> None:
        cfg = build_env_config(
            {
                "type": "metaworld",
                "task": "mt50",
                "observation_height": 96,
                "observation_width": 96,
                "unknown": "ignored",
            }
        )
        assert isinstance(cfg, MetaworldEnvConfig)
        assert cfg.task == "mt50"
        assert cfg.observation_height == 96
        assert cfg.observation_width == 96
        assert cfg.features["pixels/corner2"].shape == (96, 96, 3)
        assert cfg.features_map["pixels/corner2"] == "observation.images.corner2"

    def test_build_env_config_uses_custom_camera_name_in_feature_map(self) -> None:
        cfg = build_env_config(
            {
                "type": "metaworld",
                "task": "mt50",
                "camera_name": "corner3",
                "obs_type": "pixels",
            }
        )

        assert cfg.features["pixels/corner3"].shape == (480, 480, 3)
        assert "agent_pos" not in cfg.features
        assert cfg.features_map == {
            "action": "action",
            "pixels/corner3": "observation.images.corner3",
        }

    @pytest.mark.parametrize("episode_length", [0, -1])
    def test_build_env_config_rejects_invalid_episode_length(
        self,
        episode_length: int,
    ) -> None:
        with pytest.raises(ValueError, match="episode_length must be a positive"):
            build_env_config(
                {
                    "type": "metaworld",
                    "task": "reach-v3",
                    "episode_length": episode_length,
                }
            )

    def test_list_tasks(self) -> None:
        tasks = list_tasks({"type": "metaworld", "task": "very_hard"})
        assert tasks == [
            ("very_hard", 0),
            ("very_hard", 1),
            ("very_hard", 2),
            ("very_hard", 3),
            ("very_hard", 4),
        ]

    def test_make_env_uses_local_metaworld_builder(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        metaworld_registration = importlib.import_module(
            "praxis_eval.envs.metaworld.registration"
        )
        vector_mod = importlib.import_module("gymnasium.vector")

        class _FakeSyncVectorEnv:
            def __init__(self, env_fns) -> None:
                self.instances = [env_fn() for env_fn in env_fns]

        monkeypatch.setattr(vector_mod, "SyncVectorEnv", _FakeSyncVectorEnv)
        monkeypatch.setattr(
            metaworld_registration,
            "_make_metaworld_env_fn",
            lambda _cfg_obj, *, task_name: lambda: {"task_name": task_name},
        )

        envs = make_env({"type": "metaworld", "task": "reach-v3"}, n_envs=2)

        assert list(envs) == ["reach-v3"]
        assert list(envs["reach-v3"]) == [0]
        assert [env["task_name"] for env in envs["reach-v3"][0].instances] == [
            "reach-v3",
            "reach-v3",
        ]

    def test_eval_pool_reuses_worker_pool_per_wave(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        metaworld_eval = importlib.import_module("praxis_eval.envs.metaworld.eval")
        runtime_mod = importlib.import_module("praxis_eval.envs.metaworld.runtime")
        created_pools: list[object] = []

        class _FakeAsyncVectorEnv:
            def __init__(self, env_fns, *, dummy_env_fn) -> None:
                _ = dummy_env_fn
                self.num_envs = len(env_fns)
                self.instances = [env_fn() for env_fn in env_fns]
                self.call_each_calls: list[list[tuple[int, int, str]]] = []
                created_pools.append(self)

            def call_each(self, name, *, args_list, kwargs_list):
                assert name == "prepare_eval_job"
                self.call_each_calls.append(
                    [
                        (int(task_id), int(episode_index), str(kwargs["task_group"]))
                        for (task_id, episode_index), kwargs in zip(
                            args_list,
                            kwargs_list,
                            strict=True,
                        )
                    ]
                )
                for instance, (task_id, episode_index), kwargs in zip(
                    self.instances,
                    args_list,
                    kwargs_list,
                    strict=True,
                ):
                    instance["task_id"] = int(task_id)
                    instance["seed"] = int(episode_index)
                    instance["task_group"] = str(kwargs["task_group"])
                return tuple([None] * self.num_envs)

        def _fake_make_metaworld_env_fn(**kwargs):
            return lambda: {
                "task_name": kwargs["task_name"],
                "observation_width": kwargs["observation_width"],
                "episode_length": kwargs["episode_length"],
            }

        def _fake_construct_metaworld_eval_lane(env_fn, *, task_group, lane_idx):
            env = env_fn()
            env["task_group"] = task_group
            env["lane_idx"] = lane_idx
            return env

        monkeypatch.setattr(metaworld_eval, "AsyncVectorEnv", _FakeAsyncVectorEnv)
        monkeypatch.setattr(
            runtime_mod,
            "make_dummy_metaworld_env_fn",
            lambda **kwargs: lambda: SimpleNamespace(close=lambda: None),
        )
        monkeypatch.setattr(
            runtime_mod,
            "make_metaworld_env_fn",
            _fake_make_metaworld_env_fn,
        )
        monkeypatch.setattr(
            runtime_mod,
            "construct_metaworld_eval_lane",
            _fake_construct_metaworld_eval_lane,
        )

        cfg = MetaworldEnvConfig(
            task="mt50",
            observation_width=64,
            observation_height=64,
            episode_length=123,
        )
        handle = metaworld_eval.build_metaworld_eval_pool(
            cfg,
            tasks=list_tasks({"type": "metaworld", "task": "mt50"}),
            n_envs=2,
        )
        assert handle.env_pool is None
        assert handle.num_envs == 2

        first_jobs = [
            EvalLaneJob("easy", 0, 0, 11),
            EvalLaneJob("very_hard", 4, 0, 22),
        ]
        handle.prepare_jobs(first_jobs)
        assert handle.env_pool is created_pools[0]
        assert [env["task_name"] for env in handle.env_pool.instances] == [
            "button-press-v3",
            "pick-place-wall-v3",
        ]
        assert [env["observation_width"] for env in handle.env_pool.instances] == [
            64,
            64,
        ]
        assert [env["episode_length"] for env in handle.env_pool.instances] == [
            123,
            123,
        ]

        second_jobs = [
            EvalLaneJob("medium", 0, 1, 33),
            EvalLaneJob("hard", 5, 1, 44),
        ]
        first_pool = handle.env_pool
        handle.prepare_jobs(second_jobs)
        assert handle.env_pool is first_pool
        assert len(created_pools) == 1
        assert first_pool.call_each_calls == [[(0, 33, "medium"), (5, 44, "hard")]]
        assert [env["seed"] for env in handle.env_pool.instances] == [33, 44]


class TestDummyMetaworldEnv:
    def test_dummy_env_spaces(self) -> None:
        from praxis_eval.envs.metaworld.runtime import _DummyMetaworldEnv

        env = _DummyMetaworldEnv(
            obs_type="pixels_agent_pos",
            observation_height=64,
            observation_width=64,
        )
        observation_space = cast(spaces.Dict, env.observation_space)
        assert "pixels" in observation_space.spaces
        assert "agent_pos" in observation_space.spaces
        pixels_space = cast(spaces.Dict, observation_space["pixels"])
        assert pixels_space["corner2"].shape == (64, 64, 3)
        assert observation_space["agent_pos"].shape == (4,)
        assert env.action_space.shape == (4,)

    def test_dummy_env_uses_configured_camera_name(self) -> None:
        from praxis_eval.envs.metaworld.runtime import _DummyMetaworldEnv

        env = _DummyMetaworldEnv(
            obs_type="pixels",
            camera_name="corner3",
            observation_height=32,
            observation_width=40,
        )
        observation_space = cast(spaces.Dict, env.observation_space)
        pixels_space = cast(spaces.Dict, observation_space["pixels"])
        assert list(pixels_space.spaces) == ["corner3"]
        assert pixels_space["corner3"].shape == (32, 40, 3)

    def test_dummy_env_raises_on_reset(self) -> None:
        from praxis_eval.envs.metaworld.runtime import _DummyMetaworldEnv

        env = _DummyMetaworldEnv()
        with pytest.raises(NotImplementedError):
            env.reset()


class TestMetaworldEnv:
    @pytest.mark.parametrize("episode_length", [0, -1])
    def test_env_rejects_invalid_episode_length(self, episode_length: int) -> None:
        from praxis_eval.envs.metaworld.env import MetaworldEnv

        with pytest.raises(ValueError, match="episode_length must be a positive"):
            MetaworldEnv(task="reach-v3", episode_length=episode_length)

    def test_local_env_constructs_backend_lazily(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from praxis_eval.envs.metaworld.env import MetaworldEnv

        created: list[tuple[str, int, int]] = []

        class _FakeBackend:
            max_path_length = 321

            def __init__(
                self,
                *,
                render_mode: str,
                camera_name: str,
                width: int,
                height: int,
            ) -> None:
                _ = render_mode
                created.append((camera_name, width, height))
                self.width = width
                self.height = height
                self.model = SimpleNamespace(cam_pos=[None, None, None])
                self.seeded_rand_vec = False
                self.seed_calls: list[int] = []
                self.reset_calls = 0
                self.step_actions: list[np.ndarray] = []

            def set_task(self, task) -> None:
                self.task = task

            def seed(self, seed: int) -> list[int]:
                self.seed_calls.append(seed)
                return [seed]

            def reset(self, seed: int | None = None):
                _ = seed
                self.reset_calls += 1
                return np.array([1.0, 2.0, 3.0, 4.0]), {}

            def render(self) -> np.ndarray:
                image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                image[0, 0, 0] = 11
                image[-1, -1, 0] = 99
                return image

            def step(self, action: np.ndarray):
                self.step_actions.append(action)
                return (
                    np.array([5.0, 6.0, 7.0, 8.0]),
                    0.25,
                    False,
                    False,
                    {"success": 0},
                )

            def close(self) -> None:
                pass

        class _FakeMT1:
            def __init__(self, env_name: str, seed: int) -> None:
                _ = seed
                self.train_classes = {env_name: _FakeBackend}
                self.train_tasks = [SimpleNamespace(name=env_name)]

        fake_metaworld = cast(Any, ModuleType("metaworld"))
        fake_metaworld.MT1 = _FakeMT1
        monkeypatch.setitem(sys.modules, "metaworld", fake_metaworld)

        env = MetaworldEnv(
            task="metaworld-reach-v3", observation_width=8, observation_height=6
        )
        assert env.task == "reach-v3"
        assert env.task_description == "Reach a goal position"
        assert created == []

        obs, info = env.reset(seed=123)
        assert created == [("corner2", 8, 6)]
        assert info == {"is_success": False}
        backend = cast(Any, env._env)
        assert backend.seed_calls == [123]
        assert backend.seeded_rand_vec is True
        assert obs["pixels"]["corner2"].shape == (6, 8, 3)
        assert obs["pixels"]["corner2"][0, 0, 0] == 99
        np.testing.assert_array_equal(obs["agent_pos"], np.array([1.0, 2.0, 3.0, 4.0]))

        step_obs, reward, terminated, truncated, step_info = env.step(
            np.zeros((4,), dtype=np.float32)
        )
        assert reward == 0.25
        assert not terminated
        assert not truncated
        assert step_info["task"] == "reach-v3"
        assert step_info["truncated"] is False
        assert backend.step_actions[0].dtype == np.float32
        np.testing.assert_array_equal(
            step_obs["agent_pos"],
            np.array([5.0, 6.0, 7.0, 8.0]),
        )

    def test_vector_env_preprocesses_named_pixels_to_contract_key(self) -> None:
        import gymnasium as gym
        from gymnasium.vector import SyncVectorEnv
        from lerobot.envs.utils import preprocess_observation

        from praxis_eval.envs.metaworld.env import (
            make_action_space,
            make_observation_space,
        )

        class _MiniMetaworldEnv(gym.Env):
            metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

            def __init__(self) -> None:
                super().__init__()
                self.observation_space = make_observation_space(
                    obs_type="pixels_agent_pos",
                    camera_name="corner2",
                    observation_height=6,
                    observation_width=8,
                )
                self.action_space = make_action_space()

            def reset(self, *, seed: int | None = None, options=None):
                _ = options
                super().reset(seed=seed)
                return (
                    {
                        "pixels": {
                            "corner2": np.zeros((6, 8, 3), dtype=np.uint8),
                        },
                        "agent_pos": np.ones((4,), dtype=np.float64),
                    },
                    {},
                )

            def step(self, action):
                _ = action
                observation, _info = self.reset()
                return observation, 0.0, False, False, {}

        env = SyncVectorEnv([_MiniMetaworldEnv, _MiniMetaworldEnv])
        try:
            obs, _info = env.reset()
            processed = preprocess_observation(obs)
        finally:
            env.close()

        assert obs["pixels"]["corner2"].shape == (2, 6, 8, 3)
        assert "observation.image" not in processed
        assert processed["observation.images.corner2"].shape == (2, 3, 6, 8)
        assert processed["observation.state"].shape == (2, 4)

    def test_local_env_does_not_reset_inside_terminal_step(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from praxis_eval.envs.metaworld.env import MetaworldEnv

        class _FakeBackend:
            max_path_length = 500

            def __init__(self, **kwargs) -> None:
                _ = kwargs
                self.model = SimpleNamespace(cam_pos=[None, None, None])
                self.reset_calls = 0

            def set_task(self, task) -> None:
                self.task = task

            def reset(self, seed: int | None = None):
                _ = seed
                self.reset_calls += 1
                return np.zeros((4,), dtype=np.float64), {}

            def render(self) -> np.ndarray:
                return np.zeros((4, 4, 3), dtype=np.uint8)

            def step(self, action: np.ndarray):
                _ = action
                return (
                    np.ones((4,), dtype=np.float64),
                    1.0,
                    False,
                    False,
                    {"success": 1},
                )

            def close(self) -> None:
                pass

        class _FakeMT1:
            def __init__(self, env_name: str, seed: int) -> None:
                _ = seed
                self.train_classes = {env_name: _FakeBackend}
                self.train_tasks = [SimpleNamespace(name=env_name)]

        fake_metaworld = cast(Any, ModuleType("metaworld"))
        fake_metaworld.MT1 = _FakeMT1
        monkeypatch.setitem(sys.modules, "metaworld", fake_metaworld)

        env = MetaworldEnv(
            task="reach-v3",
            observation_width=4,
            observation_height=4,
            episode_length=1,
        )
        env.reset(seed=7)
        backend = cast(Any, env._env)
        assert backend.reset_calls == 2  # backend construction + explicit reset

        _obs, _reward, terminated, truncated, info = env.step(
            np.zeros((4,), dtype=np.float32)
        )

        assert terminated is True
        assert truncated is True
        assert info["final_info"]["is_success"] is True
        assert info["final_info"]["truncated"] is True
        assert backend.reset_calls == 2

    def test_local_env_validates_action_shape_and_bounds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from praxis_eval.envs.metaworld.env import MetaworldEnv

        class _FakeBackend:
            max_path_length = 500

            def __init__(self, **kwargs) -> None:
                _ = kwargs
                self.model = SimpleNamespace(cam_pos=[None, None, None])

            def set_task(self, task) -> None:
                self.task = task

            def reset(self, seed: int | None = None):
                _ = seed
                return np.zeros((4,), dtype=np.float64), {}

            def render(self) -> np.ndarray:
                return np.zeros((4, 4, 3), dtype=np.uint8)

            def step(self, action: np.ndarray):
                _ = action
                return np.zeros((4,), dtype=np.float64), 0.0, False, False, {}

            def close(self) -> None:
                pass

        class _FakeMT1:
            def __init__(self, env_name: str, seed: int) -> None:
                _ = seed
                self.train_classes = {env_name: _FakeBackend}
                self.train_tasks = [SimpleNamespace(name=env_name)]

        fake_metaworld = cast(Any, ModuleType("metaworld"))
        fake_metaworld.MT1 = _FakeMT1
        monkeypatch.setitem(sys.modules, "metaworld", fake_metaworld)

        env = MetaworldEnv(task="reach-v3", observation_width=4, observation_height=4)

        with pytest.raises(ValueError, match="Expected action shape"):
            env.step(np.zeros((5,), dtype=np.float32))
        with pytest.raises(ValueError, match="within the normalized action space"):
            env.step(np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32))


class TestMetaworldEvalLaneWrapper:
    def test_task_description_updates_after_retarget(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import gymnasium as gym
        from gymnasium import spaces

        from praxis_eval.envs.metaworld.runtime import MetaworldEvalLaneWrapper

        runtime_mod = importlib.import_module("praxis_eval.envs.metaworld.runtime")

        class _FakeEnv(gym.Env):
            camera_name = "corner2"
            obs_type = "pixels_agent_pos"
            render_mode = "rgb_array"
            observation_width = 64
            observation_height = 64
            visualization_width = 64
            visualization_height = 64
            _max_episode_steps = 500

            def __init__(self, task: str) -> None:
                super().__init__()
                self.task = task
                self.task_description = get_task_description(task)
                self.observation_space = spaces.Dict({})
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                )
                self.closed = False

            def close(self) -> None:
                self.closed = True

        built: list[str] = []

        def _fake_metaworld_env(**kwargs):
            built.append(str(kwargs["task"]))
            return _FakeEnv(task=str(kwargs["task"]))

        monkeypatch.setattr(
            runtime_mod,
            "MetaworldEnv",
            _fake_metaworld_env,
        )

        env = _FakeEnv("reach-v3")
        wrapper = MetaworldEvalLaneWrapper(env, task_group="easy", lane_idx=0)
        assert wrapper.task_description == "Reach a goal position"

        wrapper.prepare_eval_job(task_id=1, episode_index=7, task_group="easy")
        assert wrapper.task == "button-press-topdown-v3"
        assert wrapper.task_description == "Press a button from the top"
        assert built == ["button-press-topdown-v3"]
