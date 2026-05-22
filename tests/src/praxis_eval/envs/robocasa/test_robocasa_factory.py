"""Unit tests for RoboCasa factory registration and task helpers.

No robocasa / robosuite / OpenGL is imported here — all heavy deps are
monkeypatched out so these tests run on the login node.
"""

from __future__ import annotations

import importlib
import sys
import types
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import Any

import pytest
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from praxis_eval.envs.eval_pool import EvalLaneJob
from praxis_eval.envs.factory import (
    available_env_types,
    available_eval_pool_env_types,
    build_env_config,
    list_tasks,
)
from praxis_eval.envs.robocasa.config import RobocasaEnvConfig
from praxis_eval.envs.robocasa.tasks import (
    get_subtasks,
    get_task_horizon,
    infer_robocasa_eval_target_from_dataset,
    is_multitask,
    list_robocasa_tasks,
)

_FAKE_ROBOCASA_HORIZONS = {
    "StartCoffeeMachine": 200,
    "CloseToasterOvenDoor": 500,
    "OpenDrawer": 500,
    "PickPlaceDrawerToCounter": 500,
    "TurnOnElectricKettle": 500,
    "SlideDishwasherRack": 500,
    "CloseDrawer": 500,
}


@pytest.fixture(autouse=True)
def _stub_robocasa_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    import praxis_eval.envs.robocasa.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "_task_sets", lambda: {})
    monkeypatch.setattr(tasks_mod, "_task_horizons", lambda: _FAKE_ROBOCASA_HORIZONS)
    monkeypatch.setattr(
        tasks_mod,
        "_registered_env_names",
        lambda: tuple(_FAKE_ROBOCASA_HORIZONS),
    )
    monkeypatch.setattr(
        tasks_mod, "list_leaf_tasks", lambda: list(_FAKE_ROBOCASA_HORIZONS)
    )


def _fake_official_raw_obs(image_size: int = 64) -> dict[str, Any]:
    import numpy as np

    return {
        "robot0_base_pos": np.zeros((3,), dtype=np.float32),
        "robot0_base_quat": np.zeros((4,), dtype=np.float32),
        "robot0_base_to_eef_pos": np.zeros((3,), dtype=np.float32),
        "robot0_base_to_eef_quat": np.zeros((4,), dtype=np.float32),
        "robot0_gripper_qpos": np.zeros((2,), dtype=np.float32),
        "robot0_agentview_left_image": np.zeros(
            (image_size, image_size, 3), dtype=np.uint8
        ),
    }


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------


class TestRobocasaTasks:
    def test_is_multitask_group(self):
        assert is_multitask("mt5")
        assert not is_multitask("CloseDrawer")
        assert not is_multitask("UnknownTask")

    def test_get_subtasks_single(self):
        assert get_subtasks("CloseDrawer") == ["CloseDrawer"]

    def test_get_subtasks_mt5(self):
        subtasks = get_subtasks("mt5")
        assert subtasks == [
            "CloseToasterOvenDoor",
            "OpenDrawer",
            "PickPlaceDrawerToCounter",
            "TurnOnElectricKettle",
            "SlideDishwasherRack",
        ]

    def test_get_subtasks_dataset_task_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "praxis_eval.envs.robocasa.tasks._task_sets",
            lambda: {"atomic_seen": ("OpenDrawer", "CloseDrawer")},
        )
        assert get_subtasks("atomic_seen") == ["OpenDrawer", "CloseDrawer"]

    def test_list_robocasa_tasks_single(self):
        cfg = RobocasaEnvConfig(task="CloseDrawer")
        result = list_robocasa_tasks({}, cfg)
        assert result == [("CloseDrawer", 0)]

    def test_list_robocasa_tasks_group(self):
        cfg = RobocasaEnvConfig(task="mt5")
        result = list_robocasa_tasks({}, cfg)
        assert len(result) == 5
        names, ids = zip(*result, strict=False)
        assert list(names) == [
            "CloseToasterOvenDoor",
            "OpenDrawer",
            "PickPlaceDrawerToCounter",
            "TurnOnElectricKettle",
            "SlideDishwasherRack",
        ]
        assert list(ids) == list(range(5))

    def test_get_task_horizon_single_task(self):
        assert get_task_horizon("StartCoffeeMachine") == 200
        assert get_task_horizon("OpenDrawer") == 500

    def test_get_task_horizon_group_uses_max_subtask_horizon(self):
        assert get_task_horizon("mt5") == max(
            get_task_horizon("CloseToasterOvenDoor"),
            get_task_horizon("OpenDrawer"),
            get_task_horizon("PickPlaceDrawerToCounter"),
            get_task_horizon("TurnOnElectricKettle"),
            get_task_horizon("SlideDishwasherRack"),
        )

    def test_get_task_horizon_unknown_uses_default(self):
        assert get_task_horizon("UnknownTask") == 500
        assert get_task_horizon("UnknownTask", default=777) == 777

    def test_infer_eval_target_known_group(self):
        result = infer_robocasa_eval_target_from_dataset("robocasa_mt5")
        assert result == ("robocasa", "mt5")

    def test_infer_eval_target_single_task(self):
        result = infer_robocasa_eval_target_from_dataset("robocasa_CloseDrawer")
        assert result == ("robocasa", "CloseDrawer")

    def test_infer_eval_target_task_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "praxis_eval.envs.robocasa.tasks._task_sets",
            lambda: {"atomic_seen": ("OpenDrawer", "CloseDrawer")},
        )
        result = infer_robocasa_eval_target_from_dataset("robocasa_atomic_seen")
        assert result == ("robocasa", "atomic_seen")

    def test_infer_eval_target_unknown(self):
        assert infer_robocasa_eval_target_from_dataset("libero_10") is None
        assert infer_robocasa_eval_target_from_dataset("robocasa_UnknownXYZ") is None


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


class TestRobocasaEnvConfig:
    def test_defaults(self):
        cfg = RobocasaEnvConfig()
        assert cfg.type == "robocasa"
        assert cfg.task == "mt5"
        assert cfg.split == "all"
        assert cfg.image_size == 128
        assert not hasattr(cfg, "state_ports")
        assert "robot0_agentview_left" in cfg.camera_names

    def test_custom_task(self):
        cfg = RobocasaEnvConfig(
            task="CloseDrawer", split="target", max_episode_steps=300
        )
        assert cfg.task == "CloseDrawer"
        assert cfg.split == "target"
        assert cfg.max_episode_steps == 300

    def test_custom_image_size_and_cameras_update_feature_metadata(self):
        cfg = RobocasaEnvConfig(
            image_size=96,
            camera_names=["robot0_frontview", "robot0_wristview"],
        )

        assert cfg.features[f"{OBS_IMAGES}.robot0_frontview"].shape == (96, 96, 3)
        assert cfg.features[f"{OBS_IMAGES}.robot0_wristview"].shape == (96, 96, 3)
        assert f"{OBS_IMAGES}.robot0_agentview_left" not in cfg.features
        assert cfg.features_map[f"{OBS_IMAGES}.robot0_frontview"] == (
            f"{OBS_IMAGES}.robot0_frontview"
        )
        assert cfg.features[f"{OBS_STATE}.robot0_base_pos"].shape == (3,)
        assert cfg.features[ACTION].shape == (12,)

    def test_explicit_feature_metadata_is_preserved(self):
        features = {
            "custom.image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(10, 10, 3),
            )
        }
        features_map = {"custom.image": "policy.image"}

        cfg = RobocasaEnvConfig(
            image_size=96,
            camera_names=["robot0_frontview"],
            features=features,
            features_map=features_map,
        )

        assert cfg.features is features
        assert cfg.features_map is features_map


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------


class TestRobocasaFactoryRegistration:
    def test_robocasa_in_available_env_types(self):
        assert "robocasa" in available_env_types()

    def test_robocasa_in_available_eval_pool_types(self):
        assert "robocasa" in available_eval_pool_env_types()

    def test_build_env_config_robocasa(self):
        cfg = build_env_config(
            {"type": "robocasa", "task": "mt4", "split": "pretrain", "image_size": 64}
        )
        assert isinstance(cfg, RobocasaEnvConfig)
        assert cfg.task == "mt4"
        assert cfg.split == "pretrain"
        assert cfg.image_size == 64

    def test_build_env_config_robocasa_ignores_unknown_keys(self):
        cfg = build_env_config(
            {
                "type": "robocasa",
                "task": "mt8",
                "nonexistent_key": "ignored",
            }
        )
        assert isinstance(cfg, RobocasaEnvConfig)
        assert cfg.task == "mt8"

    def test_list_tasks_robocasa(self):
        tasks = list_tasks({"type": "robocasa", "task": "mt5"})
        assert len(tasks) == 5
        assert all(isinstance(t, tuple) and len(t) == 2 for t in tasks)

    def test_build_robocasa_eval_pool_reuses_worker_pool_per_wave(self, monkeypatch):
        robocasa_eval = importlib.import_module("praxis_eval.envs.robocasa.eval")
        runtime_mod = importlib.import_module("praxis_eval.envs.robocasa.runtime")

        created_pools: list[object] = []

        class _FakeAsyncVectorEnv:
            def __init__(
                self,
                env_fns,
                *,
                dummy_env_fn,
                shared_memory=True,
                context,
            ):
                _ = dummy_env_fn
                assert shared_memory is True
                self.num_envs = len(env_fns)
                self.context = context
                self.closed = False
                self.instances = [env_fn() for env_fn in env_fns]
                self.call_each_calls: list[list[tuple[int, int, str]]] = []
                created_pools.append(self)

            def call_each(self, name, *, args_list, kwargs_list=None, timeout=None):
                _ = (kwargs_list, timeout)
                assert name == "prepare_eval_job"
                self.call_each_calls.append(
                    [
                        (int(task_id), int(episode_index), str(task_group))
                        for task_id, episode_index, task_group in args_list
                    ]
                )
                for instance, (task_id, episode_index, task_group) in zip(
                    self.instances, args_list, strict=True
                ):
                    instance["seed"] = int(episode_index)
                    instance["task_group"] = str(task_group)
                    instance["task_id"] = int(task_id)
                return tuple([None] * self.num_envs)

            def close(self, *, terminate=False):
                _ = terminate
                self.closed = True

        def _fake_build_robocasa_env_with_retries(**kwargs):
            return SimpleNamespace(
                image_size=int(kwargs["image_size"]),
                camera_names=list(kwargs["camera_names"]),
                _max_episode_steps=int(kwargs["max_episode_steps"]),
                task_name=str(kwargs["task_name"]),
                split=str(kwargs["split"]),
                _seed=int(kwargs["seed"]),
            )

        def _fake_construct_robocasa_eval_lane(env_fn, *, task_group, lane_idx):
            env = env_fn()
            return {
                "lane_idx": lane_idx,
                "task_group": task_group,
                "task_name": env.task_name,
                "split": env.split,
                "seed": env._seed,
                "max_episode_steps": env._max_episode_steps,
            }

        monkeypatch.setattr(robocasa_eval, "AsyncVectorEnv", _FakeAsyncVectorEnv)
        monkeypatch.setattr(
            runtime_mod,
            "build_robocasa_env_with_retries",
            _fake_build_robocasa_env_with_retries,
        )
        monkeypatch.setattr(
            runtime_mod,
            "construct_robocasa_eval_lane",
            _fake_construct_robocasa_eval_lane,
        )
        monkeypatch.setattr(
            runtime_mod,
            "make_dummy_robocasa_env_fn",
            lambda **kwargs: lambda: SimpleNamespace(close=lambda: None),
        )

        cfg = RobocasaEnvConfig(task="mt5", split="target", max_episode_steps=200)
        handle = robocasa_eval.build_robocasa_eval_pool(
            cfg,
            tasks=list_tasks({"type": "robocasa", "task": "mt5"}),
            n_envs=2,
        )
        assert handle.env_pool is None
        assert handle.num_envs == 2

        first_jobs = [
            EvalLaneJob("mt5", 0, 0, 11),
            EvalLaneJob("mt5", 3, 0, 22),
        ]
        handle.prepare_jobs(first_jobs)
        assert handle.env_pool is created_pools[0]
        assert [env["task_name"] for env in handle.env_pool.instances] == [
            "CloseToasterOvenDoor",
            "TurnOnElectricKettle",
        ]
        assert [env["split"] for env in handle.env_pool.instances] == [
            "target",
            "target",
        ]
        assert [env["seed"] for env in handle.env_pool.instances] == [11, 22]

        second_jobs = [
            EvalLaneJob("mt5", 1, 1, 33),
            EvalLaneJob("mt5", 4, 1, 44),
        ]
        first_pool = handle.env_pool
        handle.prepare_jobs(second_jobs)
        assert first_pool.closed is False
        assert handle.env_pool is created_pools[0]
        assert len(created_pools) == 1
        assert first_pool.call_each_calls == [[(1, 33, "mt5"), (4, 44, "mt5")]]
        assert [env["seed"] for env in handle.env_pool.instances] == [33, 44]


# ---------------------------------------------------------------------------
# Dummy env (space inference, no sim)
# ---------------------------------------------------------------------------


class TestDummyRobocasaEnv:
    def test_dummy_env_spaces(self):
        from praxis_eval.envs.robocasa.runtime import _DummyRobocasaEnv

        env = _DummyRobocasaEnv(
            camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
            image_size=64,
        )
        # Obs space is nested: pixels dict + robot_state dict, no prompt key.
        assert "pixels" in env.observation_space.spaces
        assert "robot_state" in env.observation_space.spaces
        assert "robot0_agentview_left" in env.observation_space["pixels"].spaces
        assert "robot0_eye_in_hand" in env.observation_space["pixels"].spaces
        assert "base_pos" in env.observation_space["robot_state"].spaces
        assert "base_quat" in env.observation_space["robot_state"].spaces
        assert "base_to_eef_pos" in env.observation_space["robot_state"].spaces
        assert "base_to_eef_quat" in env.observation_space["robot_state"].spaces
        assert "gripper_qpos" in env.observation_space["robot_state"].spaces
        assert env.observation_space["pixels"]["robot0_agentview_left"].shape == (
            64,
            64,
            3,
        )
        assert env.observation_space["robot_state"]["base_pos"].shape == (3,)
        assert env.observation_space["robot_state"]["base_quat"].shape == (4,)
        assert env.observation_space["robot_state"]["base_to_eef_pos"].shape == (3,)
        assert env.observation_space["robot_state"]["base_to_eef_quat"].shape == (4,)
        assert env.observation_space["robot_state"]["gripper_qpos"].shape == (2,)
        # Action space
        assert env.action_space.shape == (12,)

    def test_dummy_env_raises_on_reset(self):
        from praxis_eval.envs.robocasa.runtime import _DummyRobocasaEnv

        env = _DummyRobocasaEnv(camera_names=["robot0_agentview_left"])
        with pytest.raises(NotImplementedError):
            env.reset()

    def test_make_dummy_robocasa_env_fn(self):
        from praxis_eval.envs.robocasa.runtime import make_dummy_robocasa_env_fn

        fn = make_dummy_robocasa_env_fn(
            camera_names=["robot0_agentview_left"],
            image_size=32,
        )
        env = fn()
        assert env.observation_space["pixels"]["robot0_agentview_left"].shape == (
            32,
            32,
            3,
        )
        env.close()


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class TestRobocasaProcessorStep:
    def test_flattens_robot_state(self):
        import torch
        from lerobot.utils.constants import OBS_PREFIX, OBS_STATE

        from praxis_eval.envs.robocasa.processor import RobocasaProcessorStep

        B = 2
        obs = {
            f"{OBS_PREFIX}robot_state": {
                "base_pos": torch.zeros(B, 3),
                "base_quat": torch.ones(B, 4),
                "base_to_eef_pos": torch.full((B, 3), 2.0),
                "base_to_eef_quat": torch.full((B, 4), 3.0),
                "gripper_qpos": torch.full((B, 2), 4.0),
            },
            "observation.images.robot0_agentview_left": torch.zeros(B, 3, 64, 64),
        }
        proc = RobocasaProcessorStep()
        out = proc.observation(obs)

        assert OBS_STATE in out
        assert out[OBS_STATE].shape == (B, 16)
        assert torch.all(out[OBS_STATE][:, :3] == 0.0)
        assert torch.all(out[OBS_STATE][:, 3:7] == 1.0)
        assert torch.all(out[OBS_STATE][:, 7:10] == 2.0)
        assert torch.all(out[OBS_STATE][:, 10:14] == 3.0)
        assert torch.all(out[OBS_STATE][:, 14:] == 4.0)
        # images pass through unchanged
        assert "observation.images.robot0_agentview_left" in out
        # Per-key aliases remain for state-port-derived PolicyIO contracts.
        assert "observation.state.robot0_base_pos" in out
        assert "observation.state.robot0_base_quat" in out
        assert "observation.state.robot0_base_to_eef_pos" in out
        assert "observation.state.robot0_base_to_eef_quat" in out
        assert "observation.state.robot0_gripper_qpos" in out
        # robot_state key removed
        assert f"{OBS_PREFIX}robot_state" not in out

    def test_missing_state_key_raises(self):
        import torch
        from lerobot.utils.constants import OBS_PREFIX

        from praxis_eval.envs.robocasa.processor import RobocasaProcessorStep

        obs = {f"{OBS_PREFIX}robot_state": {"base_pos": torch.zeros(1, 3)}}
        proc = RobocasaProcessorStep()
        with pytest.raises(KeyError, match="base_quat"):
            proc.observation(obs)

    def test_transform_features_collapses_state(self):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.utils.constants import OBS_STATE

        from praxis_eval.envs.robocasa.processor import RobocasaProcessorStep

        features = {
            FeatureType.STATE: {
                "observation.state.robot0_base_pos": PolicyFeature(
                    type=FeatureType.STATE, shape=(3,)
                ),
                "observation.state.robot0_base_quat": PolicyFeature(
                    type=FeatureType.STATE, shape=(4,)
                ),
                "observation.state.robot0_base_to_eef_pos": PolicyFeature(
                    type=FeatureType.STATE, shape=(3,)
                ),
                "observation.state.robot0_base_to_eef_quat": PolicyFeature(
                    type=FeatureType.STATE, shape=(4,)
                ),
                "observation.state.robot0_gripper_qpos": PolicyFeature(
                    type=FeatureType.STATE, shape=(2,)
                ),
            },
            FeatureType.VISUAL: {
                "observation.images.robot0_agentview_left": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 64, 64)
                ),
            },
        }
        proc = RobocasaProcessorStep()
        out = proc.transform_features(features)

        assert FeatureType.STATE in out
        assert OBS_STATE in out[FeatureType.STATE]
        assert out[FeatureType.STATE][OBS_STATE].shape == (16,)
        # Visual features pass through
        assert FeatureType.VISUAL in out

    def test_env_preprocessor_uses_official_state_order(self):
        from praxis_eval.processing import make_env_pre_post_processors

        proc, _ = make_env_pre_post_processors(
            RobocasaEnvConfig(), policy_cfg=SimpleNamespace()
        )
        step = proc.steps[0]
        assert step.state_keys == (
            "base_pos",
            "base_quat",
            "base_to_eef_pos",
            "base_to_eef_quat",
            "gripper_qpos",
        )


class TestRobocasaRuntimeRetry:
    def test_rebuild_closes_old_env_before_new_env_created(self, monkeypatch):
        import gymnasium as gym
        import numpy as np

        from praxis_eval.envs.robocasa import runtime as runtime_mod

        call_order: list[str] = []

        class _FakeEnv(gym.Env):
            def __init__(self, task_name: str):
                super().__init__()
                self.image_size = 64
                self.camera_names = ["robot0_agentview_left"]
                self._max_episode_steps = 100
                self.task_name = task_name
                self.task_description = task_name
                self.split = "target"
                self.observation_space = gym.spaces.Dict(
                    {"dummy": gym.spaces.Box(low=0.0, high=1.0, shape=(1,))}
                )
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

            def reset(self, *, seed=None, options=None):
                del seed, options
                return {"dummy": np.zeros((1,), dtype=np.float32)}, {}

            def step(self, action):
                del action
                return (
                    {"dummy": np.zeros((1,), dtype=np.float32)},
                    0.0,
                    False,
                    False,
                    {},
                )

            def close(self):
                call_order.append("close")

        def _fake_gc_collect():
            call_order.append("gc_collect")
            return 0

        def _fake_build_env_with_retries(**kwargs):
            assert kwargs["split"] == "target"
            call_order.append("build")
            return _FakeEnv(task_name="OpenDrawer")

        monkeypatch.setattr(runtime_mod.gc, "collect", _fake_gc_collect)
        monkeypatch.setattr(
            runtime_mod,
            "build_robocasa_env_with_retries",
            _fake_build_env_with_retries,
        )

        lane = runtime_mod.RobocasaEvalLaneWrapper(
            _FakeEnv(task_name="CloseDrawer"),
            task_group="mt5",
            lane_idx=0,
        )
        lane._rebuild(task_name="OpenDrawer", seed=17)

        assert call_order == ["close", "gc_collect", "build"]

    def test_prepare_eval_job_reuses_same_task_env_without_inner_reseed(self):
        from praxis_eval.envs.robocasa import runtime as runtime_mod

        class _FakeEnv(runtime_mod.gym.Env):
            def __init__(self):
                super().__init__()
                self.image_size = 64
                self.camera_names = ["robot0_agentview_left"]
                self._max_episode_steps = 100
                self.task_name = "OpenDrawer"
                self.task_description = "OpenDrawer"
                self.split = "target"
                self._seed = 11
                self.reset_calls: list[tuple[int | None, dict[str, Any] | None]] = []

            def reset(self, *, seed=None, options=None):
                self.reset_calls.append((seed, options))
                return (seed, options)

        env = _FakeEnv()
        lane = runtime_mod.RobocasaEvalLaneWrapper(
            env,
            task_group="mt5",
            lane_idx=0,
        )

        lane.prepare_eval_job(task_id=1, episode_index=22, task_group="mt5")

        assert lane.env is env
        assert lane._current_task == "OpenDrawer"
        assert lane._current_seed == 11
        assert lane.reset() == (
            None,
            {"episode_seed": 22, "reseed_inner_env": False},
        )
        assert lane._current_seed == 22
        assert env.reset_calls == [
            (None, {"episode_seed": 22, "reseed_inner_env": False}),
        ]

    def test_prepare_eval_job_rebuild_preserves_configured_split(self, monkeypatch):
        import numpy as np

        from praxis_eval.envs.robocasa import runtime as runtime_mod

        seen_builds: list[dict[str, Any]] = []

        class _FakeEnv(runtime_mod.gym.Env):
            def __init__(self, task_name: str, split: str, seed: int = 0):
                super().__init__()
                self.image_size = 64
                self.camera_names = ["robot0_agentview_left"]
                self._max_episode_steps = 100
                self.task_name = task_name
                self.task_description = task_name
                self.split = split
                self._seed = seed
                self.observation_space = runtime_mod.gym.spaces.Dict(
                    {"dummy": runtime_mod.gym.spaces.Box(0.0, 1.0, shape=(1,))}
                )
                self.action_space = runtime_mod.gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                )

            def reset(self, *, seed=None, options=None):
                _ = (seed, options)
                return {}, {}

            def close(self):
                return None

        def _fake_build_env_with_retries(**kwargs):
            seen_builds.append(dict(kwargs))
            return _FakeEnv(
                task_name=str(kwargs["task_name"]),
                split=str(kwargs["split"]),
                seed=int(kwargs["seed"]),
            )

        monkeypatch.setattr(
            runtime_mod,
            "build_robocasa_env_with_retries",
            _fake_build_env_with_retries,
        )

        lane = runtime_mod.RobocasaEvalLaneWrapper(
            _FakeEnv(task_name="CloseToasterOvenDoor", split="target", seed=11),
            task_group="mt5",
            lane_idx=0,
        )

        lane.prepare_eval_job(task_id=1, episode_index=22, task_group="mt5")

        assert seen_builds[-1]["task_name"] == "OpenDrawer"
        assert seen_builds[-1]["split"] == "target"
        assert lane.env.split == "target"

    def test_build_env_with_retries_recovers_from_retryable_value_error(
        self, monkeypatch
    ):
        from praxis_eval.envs.robocasa.runtime import build_robocasa_env_with_retries

        calls: list[int] = []

        class _FakeEnv:
            def __init__(self, **kwargs):
                calls.append(int(kwargs["seed"]))
                if len(calls) < 3:
                    raise ValueError(
                        "Error: for mesh geoms, inertia should be specified in the mesh asset"
                    )

        fake_mod = types.ModuleType("praxis_eval.envs.robocasa.env")
        fake_mod.RobocasaEnv = _FakeEnv
        monkeypatch.setitem(sys.modules, "praxis_eval.envs.robocasa.env", fake_mod)

        env = build_robocasa_env_with_retries(
            task_name="CloseDrawer",
            image_size=64,
            seed=10,
            camera_names=["robot0_agentview_left"],
            max_episode_steps=100,
            enable_render=True,
            max_attempts=4,
        )

        assert isinstance(env, _FakeEnv)
        assert calls == [10, 11, 12]

    def test_build_robocasa_env_with_retries_calls_gc_collect_between_retryable_failures(
        self, monkeypatch
    ):
        from praxis_eval.envs.robocasa import runtime as runtime_mod

        build_calls = 0
        gc_collect_calls = 0

        class _FakeEnv:
            def __init__(self, **kwargs):
                nonlocal build_calls
                _ = kwargs
                build_calls += 1
                if build_calls <= 2:
                    raise ValueError(
                        "Error: for mesh geoms, inertia should be specified in the mesh asset"
                    )

        def _fake_gc_collect():
            nonlocal gc_collect_calls
            gc_collect_calls += 1
            return 0

        fake_mod = types.ModuleType("praxis_eval.envs.robocasa.env")
        fake_mod.RobocasaEnv = _FakeEnv
        monkeypatch.setitem(sys.modules, "praxis_eval.envs.robocasa.env", fake_mod)
        monkeypatch.setattr(runtime_mod.gc, "collect", _fake_gc_collect)

        env = runtime_mod.build_robocasa_env_with_retries(
            task_name="CloseDrawer",
            image_size=64,
            seed=10,
            camera_names=["robot0_agentview_left"],
            max_episode_steps=100,
            enable_render=True,
            max_attempts=4,
        )

        assert isinstance(env, _FakeEnv)
        assert gc_collect_calls >= 2

    def test_build_env_with_retries_does_not_retry_non_retryable_error(
        self, monkeypatch
    ):
        from praxis_eval.envs.robocasa.runtime import build_robocasa_env_with_retries

        calls: list[int] = []

        class _FakeEnv:
            def __init__(self, **kwargs):
                calls.append(int(kwargs["seed"]))
                raise RuntimeError("boom")

        fake_mod = types.ModuleType("praxis_eval.envs.robocasa.env")
        fake_mod.RobocasaEnv = _FakeEnv
        monkeypatch.setitem(sys.modules, "praxis_eval.envs.robocasa.env", fake_mod)

        with pytest.raises(RuntimeError, match="boom"):
            build_robocasa_env_with_retries(
                task_name="CloseDrawer",
                image_size=64,
                seed=10,
                camera_names=["robot0_agentview_left"],
                max_episode_steps=100,
                enable_render=True,
                max_attempts=4,
            )
        assert calls == [10]


class TestRobocasaEvalPoolContext:
    def test_default_context_prefers_fork(self, monkeypatch):
        from praxis_eval.envs.robocasa import eval as eval_mod

        monkeypatch.setattr(
            eval_mod.multiprocessing,
            "get_all_start_methods",
            lambda: ["fork", "spawn"],
        )
        assert eval_mod._resolve_robocasa_mp_context() == "fork"

    def test_missing_fork_falls_back_to_spawn(self, monkeypatch):
        from praxis_eval.envs.robocasa import eval as eval_mod

        monkeypatch.setattr(
            eval_mod.multiprocessing,
            "get_all_start_methods",
            lambda: ["spawn"],
        )
        assert eval_mod._resolve_robocasa_mp_context() == "spawn"


class TestRobocasaEnvResetRetry:
    def test_lerobot_action_order_is_translated_before_env_step(self) -> None:
        import numpy as np

        from praxis_eval.envs.robocasa.env import RobocasaEnv

        class _FakeRawEnv:
            def __init__(self) -> None:
                self.timestep = 0
                self.step_actions: list[np.ndarray] = []

            def step(self, action: np.ndarray):
                self.step_actions.append(np.asarray(action, dtype=np.float32).copy())
                return {"raw": True}, 0.0, False, {}

            def _check_success(self) -> bool:
                return False

        raw_env = _FakeRawEnv()
        env = RobocasaEnv.__new__(RobocasaEnv)
        env.env = raw_env
        env.task_name = "OpenDrawer"
        env._step_count = 0
        env._max_episode_steps = 500
        env._done = False
        env._extract_obs = lambda raw_obs: raw_obs

        action = np.arange(12, dtype=np.float32)
        obs, reward, done, truncated, info = RobocasaEnv.step(env, action)

        assert len(raw_env.step_actions) == 1
        np.testing.assert_array_equal(
            raw_env.step_actions[0],
            np.array([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4], dtype=np.float32),
        )
        assert obs == {"raw": True}
        assert reward == 0.0
        assert done is False
        assert truncated is False
        assert info["is_success"] is False

    def test_step_rejects_non_robocasa_action_shape(self) -> None:
        import numpy as np

        from praxis_eval.envs.robocasa.env import RobocasaEnv

        env = RobocasaEnv.__new__(RobocasaEnv)
        env._step_count = 0

        with pytest.raises(ValueError, match=r"shape \(12,\)"):
            RobocasaEnv.step(env, np.zeros(7, dtype=np.float32))

    def test_action_space_is_stable_and_step_reorders_before_native_clipping(
        self, monkeypatch
    ) -> None:
        import numpy as np

        from praxis_eval.envs.robocasa.env import RobocasaEnv

        low_native = np.zeros(12, dtype=np.float32)
        high_native = np.arange(12, dtype=np.float32)

        class _FakeRawEnv:
            def __init__(self) -> None:
                self.action_dim = 12
                self.action_spec = (low_native, high_native)
                self.timestep = 0
                self.step_actions: list[np.ndarray] = []

            def _get_observations(self):
                return _fake_official_raw_obs()

            def get_ep_meta(self):
                return {"lang": "dummy"}

            def reset(self):
                return self._get_observations()

            def step(self, action: np.ndarray):
                self.step_actions.append(np.asarray(action, dtype=np.float32).copy())
                return self._get_observations(), 0.0, False, {}

            def _check_success(self) -> bool:
                return False

            def close(self):
                return None

        def _fake_create_env(**kwargs):
            _ = kwargs
            return _FakeRawEnv()

        fake_env_utils = types.ModuleType("robocasa.utils.env_utils")
        fake_env_utils.create_env = _fake_create_env
        monkeypatch.setitem(sys.modules, "robocasa.utils.env_utils", fake_env_utils)

        env = RobocasaEnv(
            task_name="CloseDrawer",
            image_size=64,
            seed=5,
            camera_names=["robot0_agentview_left"],
            max_episode_steps=50,
            enable_render=True,
        )
        try:
            assert env.action_space.shape == (12,)
            assert np.isneginf(env.action_space.low).all()
            assert np.isposinf(env.action_space.high).all()

            env.step(np.arange(12, dtype=np.float32))
            np.testing.assert_array_equal(
                env.env.step_actions[-1],
                np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4], dtype=np.float32),
            )
        finally:
            env.close()

    def test_neutralize_oversized_objaverse_visual_meshes(self, tmp_path):
        from praxis_eval.envs.robocasa.env import (
            _neutralize_oversized_objaverse_visual_meshes,
        )

        visual_dir = (
            tmp_path / "models" / "assets" / "objects" / "objaverse" / "cake" / "visual"
        )
        visual_dir.mkdir(parents=True, exist_ok=True)
        big_mesh = visual_dir / "big.obj"
        small_mesh = visual_dir / "small.obj"
        other_mesh = tmp_path / "other.obj"
        big_mesh.write_bytes(b"x" * 2048)
        small_mesh.write_bytes(b"x" * 32)
        other_mesh.write_bytes(b"x" * 4096)

        xml_in = (
            "<mujoco><asset>"
            f"<mesh name='drawer_obj_model_normalized_0_vis' file='{big_mesh}'/>"
            f"<mesh name='drawer_obj_model_normalized_1_vis' file='{small_mesh}'/>"
            f"<mesh name='robot0_link0_vis' file='{other_mesh}'/>"
            "</asset><worldbody>"
            "<geom name='g_big' type='mesh' mesh='drawer_obj_model_normalized_0_vis'/>"
            "<geom name='g_small' type='mesh' mesh='drawer_obj_model_normalized_1_vis'/>"
            "<geom name='g_other' type='mesh' mesh='robot0_link0_vis'/>"
            "</worldbody></mujoco>"
        )

        xml_out, neutralized = _neutralize_oversized_objaverse_visual_meshes(
            xml_in,
            max_mesh_bytes=256,
        )

        assert neutralized == ["drawer_obj_model_normalized_0_vis"]

        root = ET.fromstring(xml_out)
        asset = root.find("asset")
        assert asset is not None
        remaining_mesh_names = [m.get("name") for m in asset.findall("mesh")]
        assert "drawer_obj_model_normalized_0_vis" not in remaining_mesh_names
        assert "drawer_obj_model_normalized_1_vis" in remaining_mesh_names
        assert "robot0_link0_vis" in remaining_mesh_names

        geom_by_name = {g.get("name"): g for g in root.findall("./worldbody/geom")}
        assert geom_by_name["g_big"].get("type") == "box"
        assert geom_by_name["g_big"].get("mesh") is None
        assert (
            geom_by_name["g_small"].get("mesh") == "drawer_obj_model_normalized_1_vis"
        )
        assert geom_by_name["g_other"].get("mesh") == "robot0_link0_vis"

    def test_reset_rebuilds_env_on_retryable_layout_error(self, monkeypatch):
        from praxis_eval.envs.robocasa.env import RobocasaEnv

        created_seeds: list[int] = []

        class _FakeRawEnv:
            def __init__(self, seed: int):
                self.seed = int(seed)
                self.action_dim = 12
                self._reset_calls = 0

            def _get_observations(self):
                return _fake_official_raw_obs()

            def get_ep_meta(self):
                return {"lang": "dummy"}

            def reset(self):
                self._reset_calls += 1
                # First env (seed=5): init reset succeeds, next reset fails.
                if self.seed == 5 and self._reset_calls >= 2:
                    raise ValueError(
                        "Error: for mesh geoms, inertia should be specified in the mesh asset"
                    )
                return self._get_observations()

            def step(self, action):
                del action
                return self._get_observations(), 0.0, False, {}

            def _check_success(self):
                return False

            def close(self):
                return None

        def _fake_create_env(**kwargs):
            seed = int(kwargs["seed"])
            created_seeds.append(seed)
            return _FakeRawEnv(seed=seed)

        fake_env_utils = types.ModuleType("robocasa.utils.env_utils")
        fake_env_utils.create_env = _fake_create_env
        monkeypatch.setitem(sys.modules, "robocasa.utils.env_utils", fake_env_utils)

        env = RobocasaEnv(
            task_name="CloseDrawer",
            split="target",
            image_size=64,
            seed=5,
            camera_names=["robot0_agentview_left"],
            max_episode_steps=50,
            enable_render=True,
        )
        obs, _info = env.reset(seed=5)
        env.close()

        assert "pixels" in obs
        assert created_seeds == [5, 6]

    def test_reset_reseeds_inner_env_without_rebuild(self, monkeypatch):
        import numpy as np

        from praxis_eval.envs.robocasa.env import RobocasaEnv

        class _FakeRawEnv:
            def __init__(self, seed: int):
                self.seed = int(seed)
                self.action_dim = 12
                self.hard_reset = True
                self.unset_ep_meta_calls = 0

            def _get_observations(self):
                return _fake_official_raw_obs()

            def get_ep_meta(self):
                return {"lang": f"seed-{self.seed}"}

            def reset(self):
                return self._get_observations()

            def step(self, action):
                del action
                return self._get_observations(), 0.0, False, {}

            def _check_success(self):
                return False

            def close(self):
                return None

            def unset_ep_meta(self):
                self.unset_ep_meta_calls += 1

        def _fake_create_env(**kwargs):
            return _FakeRawEnv(seed=int(kwargs["seed"]))

        fake_env_utils = types.ModuleType("robocasa.utils.env_utils")
        fake_env_utils.create_env = _fake_create_env
        monkeypatch.setitem(sys.modules, "robocasa.utils.env_utils", fake_env_utils)

        env = RobocasaEnv(
            task_name="CloseDrawer",
            split="target",
            image_size=64,
            seed=5,
            camera_names=["robot0_agentview_left"],
            max_episode_steps=50,
            enable_render=True,
        )

        inner_env = env.env
        inner_env.unset_ep_meta_calls = 0
        obs, _info = env.reset(seed=9)

        assert "pixels" in obs
        assert env._seed == 9
        assert env._step_count == 0
        assert env.task_description == "seed-9"
        assert inner_env.seed == 9
        assert inner_env.unset_ep_meta_calls == 1
        assert inner_env.rng.integers(0, 1000) == np.random.default_rng(9).integers(
            0, 1000
        )

    def test_reset_can_skip_inner_reseed_when_requested(self, monkeypatch):
        import numpy as np

        from praxis_eval.envs.robocasa.env import RobocasaEnv

        class _FakeRawEnv:
            def __init__(self, seed: int):
                self.seed = int(seed)
                self.rng = np.random.default_rng(int(seed))
                self.action_dim = 12
                self.hard_reset = True
                self.unset_ep_meta_calls = 0

            def _get_observations(self):
                return _fake_official_raw_obs()

            def get_ep_meta(self):
                return {"lang": "dummy"}

            def reset(self):
                return self._get_observations()

            def step(self, action):
                del action
                return self._get_observations(), 0.0, False, {}

            def _check_success(self):
                return False

            def close(self):
                return None

            def unset_ep_meta(self):
                self.unset_ep_meta_calls += 1

        def _fake_create_env(**kwargs):
            return _FakeRawEnv(seed=int(kwargs["seed"]))

        fake_env_utils = types.ModuleType("robocasa.utils.env_utils")
        fake_env_utils.create_env = _fake_create_env
        monkeypatch.setitem(sys.modules, "robocasa.utils.env_utils", fake_env_utils)

        env = RobocasaEnv(
            task_name="CloseDrawer",
            split="target",
            image_size=64,
            seed=5,
            camera_names=["robot0_agentview_left"],
            max_episode_steps=50,
            enable_render=True,
        )

        inner_env = env.env
        original_rng = inner_env.rng
        obs, _info = env.reset(options={"episode_seed": 9, "reseed_inner_env": False})

        assert "pixels" in obs
        assert env._seed == 9
        assert env._step_count == 0
        assert env.task_description == "dummy"
        assert inner_env.seed == 5
        assert inner_env.unset_ep_meta_calls == 0
        assert inner_env.rng is original_rng

    def test_reset_rebuild_retries_create_env_retryable_errors(self, monkeypatch):
        from praxis_eval.envs.robocasa.env import RobocasaEnv

        create_calls = 0
        rebuild_seed_attempts: list[int] = []

        class _FakeRawEnv:
            def __init__(self, seed: int):
                self.seed = int(seed)
                self.action_dim = 12
                self._reset_calls = 0

            def _get_observations(self):
                return _fake_official_raw_obs()

            def get_ep_meta(self):
                return {"lang": "dummy"}

            def reset(self):
                self._reset_calls += 1
                if self.seed == 7 and self._reset_calls >= 2:
                    raise ValueError(
                        "Error: for mesh geoms, inertia should be specified in the mesh asset"
                    )
                return self._get_observations()

            def step(self, action):
                del action
                return self._get_observations(), 0.0, False, {}

            def _check_success(self):
                return False

            def close(self):
                return None

        def _fake_create_env(**kwargs):
            nonlocal create_calls
            seed = int(kwargs["seed"])
            create_calls += 1
            # First call is constructor seed=7 and must succeed.
            # Next three rebuild attempts fail retryably, fourth succeeds.
            if create_calls >= 2:
                rebuild_seed_attempts.append(seed)
                if len(rebuild_seed_attempts) <= 3:
                    raise ValueError(
                        "Error: for mesh geoms, inertia should be specified in the mesh asset"
                    )
            return _FakeRawEnv(seed=seed)

        fake_env_utils = types.ModuleType("robocasa.utils.env_utils")
        fake_env_utils.create_env = _fake_create_env
        monkeypatch.setitem(sys.modules, "robocasa.utils.env_utils", fake_env_utils)

        env = RobocasaEnv(
            task_name="CloseDrawer",
            image_size=64,
            seed=7,
            camera_names=["robot0_agentview_left"],
            max_episode_steps=50,
            enable_render=True,
        )
        obs, _info = env.reset(seed=7)
        env.close()

        assert "pixels" in obs
        assert rebuild_seed_attempts == [8, 9, 10, 11]

    def test_create_env_uses_robocasa365_render_onscreen_signature(self, monkeypatch):
        from praxis_eval.envs.robocasa.env import RobocasaEnv

        seen: dict[str, Any] = {}

        class _FakeRawEnv:
            def __init__(self, seed: int):
                self.seed = int(seed)
                self.action_dim = 12

            def _get_observations(self):
                return _fake_official_raw_obs()

            def get_ep_meta(self):
                return {"lang": "dummy"}

            def reset(self):
                return self._get_observations()

            def close(self):
                return None

        def _fake_create_env(
            env_name,
            robots="PandaOmron",
            camera_names=None,
            camera_widths=128,
            camera_heights=128,
            seed=None,
            render_onscreen=False,
            **kwargs,
        ):
            seen["env_name"] = env_name
            seen["robots"] = robots
            seen["camera_names"] = list(camera_names or [])
            seen["camera_widths"] = camera_widths
            seen["camera_heights"] = camera_heights
            seen["seed"] = seed
            seen["render_onscreen"] = render_onscreen
            seen["kwargs"] = dict(kwargs)
            return _FakeRawEnv(seed=int(seed))

        fake_env_utils = types.ModuleType("robocasa.utils.env_utils")
        fake_env_utils.create_env = _fake_create_env
        monkeypatch.setitem(sys.modules, "robocasa.utils.env_utils", fake_env_utils)

        env = RobocasaEnv(
            task_name="CloseDrawer",
            split="target",
            image_size=64,
            seed=5,
            camera_names=["robot0_agentview_left"],
            max_episode_steps=50,
            enable_render=True,
        )
        env.close()

        assert seen["env_name"] == "CloseDrawer"
        assert seen["robots"] == "PandaOmron"
        assert seen["camera_names"] == ["robot0_agentview_left"]
        assert seen["camera_widths"] == 64
        assert seen["camera_heights"] == 64
        assert seen["seed"] == 5
        assert seen["kwargs"]["split"] == "target"
        assert seen["render_onscreen"] is False
        assert "has_offscreen_renderer" not in seen["kwargs"]
        assert "use_camera_obs" not in seen["kwargs"]

    def test_create_env_keeps_legacy_renderer_kwargs_when_new_signature_missing(
        self, monkeypatch
    ):
        from praxis_eval.envs.robocasa.env import RobocasaEnv

        seen: dict[str, Any] = {}

        class _FakeRawEnv:
            def __init__(self, seed: int):
                self.seed = int(seed)
                self.action_dim = 12

            def _get_observations(self):
                return _fake_official_raw_obs()

            def get_ep_meta(self):
                return {"lang": "dummy"}

            def reset(self):
                return self._get_observations()

            def close(self):
                return None

        def _fake_create_env(**kwargs):
            seen.update(kwargs)
            return _FakeRawEnv(seed=int(kwargs["seed"]))

        fake_env_utils = types.ModuleType("robocasa.utils.env_utils")
        fake_env_utils.create_env = _fake_create_env
        monkeypatch.setitem(sys.modules, "robocasa.utils.env_utils", fake_env_utils)

        env = RobocasaEnv(
            task_name="CloseDrawer",
            split="pretrain",
            image_size=64,
            seed=5,
            camera_names=["robot0_agentview_left"],
            max_episode_steps=50,
            enable_render=True,
        )
        env.close()

        assert seen["has_onscreen_renderer"] is False
        assert seen["has_offscreen_renderer"] is True
        assert seen["use_camera_obs"] is True
        assert seen["split"] == "pretrain"


class TestRobocasaXmlCompatibility:
    def test_rewrite_shell_mesh_inertia_to_legacy(self):
        from praxis_eval.envs.robocasa.env import _rewrite_shell_mesh_inertia_to_legacy

        xml_in = """
        <mujoco>
          <asset>
            <mesh name="m0" file="/tmp/m0.obj" inertia="shell" />
            <mesh name="m1" file="/tmp/m1.obj" inertia="legacy" />
            <mesh name="m2" file="/tmp/m2.obj" />
          </asset>
        </mujoco>
        """
        xml_out, rewritten = _rewrite_shell_mesh_inertia_to_legacy(xml_in)
        root = ET.fromstring(xml_out)
        meshes = {
            mesh.get("name"): mesh.get("inertia")
            for mesh in root.findall(".//asset/mesh")
        }

        assert rewritten == 1
        assert meshes["m0"] == "legacy"
        assert meshes["m1"] == "legacy"
        assert meshes["m2"] is None

    def test_requires_legacy_mesh_inertia_keyword(self, monkeypatch):
        import types

        from praxis_eval.envs.robocasa import env as robocasa_env

        robocasa_env._requires_legacy_mesh_inertia_keyword.cache_clear()
        fake_mujoco = types.SimpleNamespace(__version__="3.2.6")
        monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
        assert robocasa_env._requires_legacy_mesh_inertia_keyword() is True

        robocasa_env._requires_legacy_mesh_inertia_keyword.cache_clear()
        fake_mujoco = types.SimpleNamespace(__version__="3.5.0")
        monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
        assert robocasa_env._requires_legacy_mesh_inertia_keyword() is False
