"""Unit tests for RoboMimic factory registration and async eval helpers."""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from praxis_eval.envs.eval_pool import EvalLaneJob
from praxis_eval.envs.factory import (
    available_env_types,
    available_eval_pool_env_types,
    build_env_config,
    infer_eval_env_target,
    list_tasks,
)
from praxis_eval.envs.robomimic.config import RobomimicEnvConfig
from praxis_eval.envs.robomimic.env import (
    _configure_default_egl_device_for_uuid_cuda_visibility,
)
from praxis_eval.envs.robomimic.tasks import (
    canonicalize_task_name,
    get_subtasks,
    get_task_horizon,
    get_task_instruction,
    infer_robomimic_eval_target_from_dataset,
    is_multitask,
    list_robomimic_tasks,
)


def _fake_raw_obs(image_size: int = 64) -> dict[str, Any]:
    return {
        "robot0_joint_pos": np.zeros((7,), dtype=np.float32),
        "robot0_eef_pos": np.zeros((3,), dtype=np.float32),
        "robot0_eef_quat": np.zeros((4,), dtype=np.float32),
        "robot0_gripper_qpos": np.zeros((2,), dtype=np.float32),
        "agentview_image": np.zeros((image_size, image_size, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros(
            (image_size, image_size, 3),
            dtype=np.uint8,
        ),
    }


class TestRobomimicTasks:
    def test_canonicalizes_common_dataset_aliases(self) -> None:
        assert canonicalize_task_name("lift") == "Lift"
        assert canonicalize_task_name("can") == "PickPlaceCan"
        assert canonicalize_task_name("square") == "NutAssemblySquare"
        assert canonicalize_task_name("tool_hang") == "ToolHang"

    def test_is_multitask_group(self) -> None:
        assert is_multitask("mt3")
        assert is_multitask("mt4")
        assert not is_multitask("Lift")

    def test_get_subtasks_group_and_leaf(self) -> None:
        assert get_subtasks("mt3") == [
            "Lift",
            "PickPlaceCan",
            "NutAssemblySquare",
        ]
        assert get_subtasks("Lift") == ["Lift"]

    def test_get_task_horizon_single_task(self) -> None:
        assert get_task_horizon("Lift") == 100
        assert get_task_horizon("can") == 200
        assert get_task_horizon("square") == 300
        assert get_task_horizon("tool_hang") == 800

    def test_get_task_horizon_group_uses_max_subtask_horizon(self) -> None:
        assert get_task_horizon("mt3") == 300
        assert get_task_horizon("mt4") == 800

    def test_get_task_horizon_unknown_uses_default(self) -> None:
        assert get_task_horizon("UnknownTask") == 800
        assert get_task_horizon("UnknownTask", default=777) == 777

    def test_get_task_instruction(self) -> None:
        assert get_task_instruction("Lift") == "Lift the cube."
        assert (
            get_task_instruction("can")
            == "Pick up the can and place it in the target bin."
        )
        assert (
            get_task_instruction("square")
            == "Fit the square nut onto its matching peg."
        )
        assert (
            get_task_instruction("tool_hang")
            == "Assemble the stand and hang the wrench on it."
        )

    def test_list_robomimic_tasks(self) -> None:
        cfg = RobomimicEnvConfig(task="mt4")
        assert list_robomimic_tasks({}, cfg) == [
            ("Lift", 0),
            ("PickPlaceCan", 1),
            ("ToolHang", 2),
            ("NutAssemblySquare", 3),
        ]

    def test_infer_eval_target(self) -> None:
        assert infer_robomimic_eval_target_from_dataset("robomimic_mt3") == (
            "robomimic",
            "mt3",
        )
        assert infer_robomimic_eval_target_from_dataset("robomimic_mt4") == (
            "robomimic",
            "mt4",
        )
        assert infer_robomimic_eval_target_from_dataset("robomimic_mt4_ph") == (
            "robomimic",
            "mt4",
        )
        assert infer_robomimic_eval_target_from_dataset("robomimic_can") == (
            "robomimic",
            "PickPlaceCan",
        )
        assert infer_robomimic_eval_target_from_dataset("libero_10") is None
        assert infer_eval_env_target("robomimic_square") == (
            "robomimic",
            "NutAssemblySquare",
        )


class TestRobomimicEnvConfig:
    def test_custom_image_size_cameras_and_state_ports_update_feature_metadata(
        self,
    ) -> None:
        cfg = RobomimicEnvConfig(
            image_size=96,
            camera_names=["frontview"],
            state_ports=["robot0_joint_pos", "robot0_gripper_qpos"],
        )

        assert cfg.features[f"{OBS_IMAGES}.frontview"].shape == (96, 96, 3)
        assert f"{OBS_IMAGES}.agentview" not in cfg.features
        assert cfg.features[f"{OBS_STATE}.robot0_joint_pos"].shape == (7,)
        assert cfg.features[f"{OBS_STATE}.robot0_gripper_qpos"].shape == (2,)
        assert f"{OBS_STATE}.robot0_eef_quat" not in cfg.features
        assert cfg.features_map[f"{OBS_IMAGES}.frontview"] == (
            f"{OBS_IMAGES}.frontview"
        )
        assert cfg.features_map[f"{OBS_STATE}.robot0_joint_pos"] == (
            f"{OBS_STATE}.robot0_joint_pos"
        )
        assert cfg.features[ACTION].shape == (7,)

    def test_explicit_feature_metadata_is_preserved(self) -> None:
        features = {
            "custom.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(5,),
            )
        }
        features_map = {"custom.state": "policy.state"}

        cfg = RobomimicEnvConfig(
            image_size=96,
            camera_names=["frontview"],
            state_ports=["robot0_joint_pos"],
            features=features,
            features_map=features_map,
        )

        assert cfg.features is features
        assert cfg.features_map is features_map


class TestRobomimicEglConfig:
    def test_skips_default_egl_device_for_mig_uuid_visibility(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
        monkeypatch.setenv(
            "CUDA_VISIBLE_DEVICES",
            "MIG-c09c7c27-9b0d-56f0-abdb-5d89f9c5b36c",
        )

        _configure_default_egl_device_for_uuid_cuda_visibility()

        assert "MUJOCO_EGL_DEVICE_ID" not in os.environ

    def test_sets_default_egl_device_for_numeric_cuda_visibility(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")

        _configure_default_egl_device_for_uuid_cuda_visibility()

        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "1"

    def test_keeps_explicit_egl_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-c09c7c27")
        monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "2")

        _configure_default_egl_device_for_uuid_cuda_visibility()

        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "2"

    def test_skips_egl_device_for_osmesa(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-c09c7c27")
        monkeypatch.setenv("MUJOCO_GL", "osmesa")

        _configure_default_egl_device_for_uuid_cuda_visibility()

        assert "MUJOCO_EGL_DEVICE_ID" not in os.environ


class TestRobomimicFactoryRegistration:
    def test_available_env_types(self) -> None:
        assert "robomimic" in available_env_types()
        assert "robomimic" in available_eval_pool_env_types()

    def test_build_env_config(self) -> None:
        cfg = build_env_config(
            {
                "type": "robomimic",
                "task": "mt4",
                "image_size": 96,
                "state_ports": ["robot0_joint_pos", "robot0_eef_pos"],
                "unknown": "ignored",
            }
        )
        assert isinstance(cfg, RobomimicEnvConfig)
        assert cfg.task == "mt4"
        assert cfg.image_size == 96
        assert cfg.state_ports == ["robot0_joint_pos", "robot0_eef_pos"]

    def test_list_tasks(self) -> None:
        tasks = list_tasks({"type": "robomimic", "task": "mt3"})
        assert tasks == [
            ("Lift", 0),
            ("PickPlaceCan", 1),
            ("NutAssemblySquare", 2),
        ]

    def test_eval_pool_reuses_worker_pool_per_wave(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        robomimic_eval = importlib.import_module("praxis_eval.envs.robomimic.eval")
        runtime_mod = importlib.import_module("praxis_eval.envs.robomimic.runtime")
        created_pools: list[object] = []

        class _FakeAsyncVectorEnv:
            def __init__(
                self,
                env_fns,
                *,
                dummy_env_fn,
                shared_memory=True,
                context,
            ) -> None:
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
                    self.instances,
                    args_list,
                    strict=True,
                ):
                    instance["seed"] = int(episode_index)
                    instance["task_group"] = str(task_group)
                    instance["task_id"] = int(task_id)
                return tuple([None] * self.num_envs)

            def close(self, *, terminate=False):
                _ = terminate
                self.closed = True

        def _fake_build_robomimic_env_with_retries(**kwargs):
            return SimpleNamespace(
                image_size=int(kwargs["image_size"]),
                camera_names=list(kwargs["camera_names"]),
                state_ports=list(kwargs["state_ports"]),
                video_camera=str(kwargs["video_camera"]),
                video_resolution=int(kwargs["video_resolution"]),
                _max_episode_steps=int(kwargs["max_episode_steps"]),
                task_name=str(kwargs["task_name"]),
                robot=str(kwargs["robot"]),
                _seed=int(kwargs["seed"]),
            )

        def _fake_construct_robomimic_eval_lane(env_fn, *, task_group, lane_idx):
            env = env_fn()
            return {
                "lane_idx": lane_idx,
                "task_group": task_group,
                "task_name": env.task_name,
                "seed": env._seed,
                "max_episode_steps": env._max_episode_steps,
            }

        monkeypatch.setattr(robomimic_eval, "AsyncVectorEnv", _FakeAsyncVectorEnv)
        monkeypatch.setattr(
            runtime_mod,
            "build_robomimic_env_with_retries",
            _fake_build_robomimic_env_with_retries,
        )
        monkeypatch.setattr(
            runtime_mod,
            "construct_robomimic_eval_lane",
            _fake_construct_robomimic_eval_lane,
        )
        monkeypatch.setattr(
            runtime_mod,
            "make_dummy_robomimic_env_fn",
            lambda **kwargs: lambda: SimpleNamespace(close=lambda: None),
        )

        cfg = RobomimicEnvConfig(task="mt3", max_episode_steps=200)
        handle = robomimic_eval.build_robomimic_eval_pool(
            cfg,
            tasks=list_tasks({"type": "robomimic", "task": "mt3"}),
            n_envs=2,
        )
        assert handle.env_pool is None
        assert handle.num_envs == 2

        first_jobs = [
            EvalLaneJob("mt3", 0, 0, 11),
            EvalLaneJob("mt3", 2, 0, 22),
        ]
        handle.prepare_jobs(first_jobs)
        assert handle.env_pool is created_pools[0]
        assert [env["task_name"] for env in handle.env_pool.instances] == [
            "Lift",
            "NutAssemblySquare",
        ]
        assert [env["max_episode_steps"] for env in handle.env_pool.instances] == [
            100,
            300,
        ]
        assert [env["seed"] for env in handle.env_pool.instances] == [11, 22]

        second_jobs = [
            EvalLaneJob("mt3", 1, 1, 33),
            EvalLaneJob("mt3", 0, 1, 44),
        ]
        first_pool = handle.env_pool
        handle.prepare_jobs(second_jobs)
        assert first_pool.closed is False
        assert handle.env_pool is created_pools[0]
        assert len(created_pools) == 1
        assert first_pool.call_each_calls == [[(1, 33, "mt3"), (0, 44, "mt3")]]
        assert [env["seed"] for env in handle.env_pool.instances] == [33, 44]

    def test_eval_pool_closes_failed_pool_on_retarget_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        robomimic_eval = importlib.import_module("praxis_eval.envs.robomimic.eval")
        runtime_mod = importlib.import_module("praxis_eval.envs.robomimic.runtime")

        class _FailingAsyncVectorEnv:
            def __init__(
                self,
                env_fns,
                *,
                dummy_env_fn,
                shared_memory=True,
                context,
            ) -> None:
                _ = (env_fns, dummy_env_fn, shared_memory, context)
                self.num_envs = 1
                self.closed_with_terminate: bool | None = None

            def call_each(self, name, *, args_list, kwargs_list=None, timeout=None):
                _ = (name, args_list, kwargs_list, timeout)
                raise RuntimeError("retarget failed")

            def close(self, *, terminate=False):
                self.closed_with_terminate = bool(terminate)

        monkeypatch.setattr(robomimic_eval, "AsyncVectorEnv", _FailingAsyncVectorEnv)
        monkeypatch.setattr(
            runtime_mod,
            "make_dummy_robomimic_env_fn",
            lambda **kwargs: lambda: SimpleNamespace(close=lambda: None),
        )
        monkeypatch.setattr(
            runtime_mod,
            "build_robomimic_env_with_retries",
            lambda **kwargs: SimpleNamespace(
                image_size=int(kwargs["image_size"]),
                camera_names=list(kwargs["camera_names"]),
                state_ports=list(kwargs["state_ports"]),
                video_camera=str(kwargs["video_camera"]),
                video_resolution=int(kwargs["video_resolution"]),
                _max_episode_steps=int(kwargs["max_episode_steps"]),
                task_name=str(kwargs["task_name"]),
                robot=str(kwargs["robot"]),
                _seed=int(kwargs["seed"]),
            ),
        )
        monkeypatch.setattr(
            runtime_mod,
            "construct_robomimic_eval_lane",
            lambda env_fn, *, task_group, lane_idx: env_fn(),
        )

        handle = robomimic_eval.build_robomimic_eval_pool(
            RobomimicEnvConfig(task="Lift"),
            tasks=[("Lift", 0)],
            n_envs=1,
        )
        handle.prepare_jobs([EvalLaneJob("Lift", 0, 0, 1)])
        first_pool = handle.env_pool

        with pytest.raises(RuntimeError, match="retarget failed"):
            handle.prepare_jobs([EvalLaneJob("Lift", 0, 1, 2)])

        assert first_pool.closed_with_terminate is True
        assert handle.env_pool is None


class TestDummyRobomimicEnv:
    def test_dummy_env_spaces(self) -> None:
        from praxis_eval.envs.robomimic.runtime import _DummyRobomimicEnv

        env = _DummyRobomimicEnv(
            camera_names=["agentview", "robot0_eye_in_hand"],
            image_size=64,
        )
        assert "pixels" in env.observation_space.spaces
        assert "robot_state" in env.observation_space.spaces
        assert "agentview" in env.observation_space["pixels"].spaces
        assert "robot0_eye_in_hand" in env.observation_space["pixels"].spaces
        assert "robot0_eef_pos" in env.observation_space["robot_state"].spaces
        assert "robot0_eef_quat" in env.observation_space["robot_state"].spaces
        assert "robot0_gripper_qpos" in env.observation_space["robot_state"].spaces
        assert env.observation_space["pixels"]["agentview"].shape == (64, 64, 3)
        assert env.observation_space["robot_state"]["robot0_eef_pos"].shape == (3,)
        assert env.action_space.shape == (7,)
        env.close()

    def test_dummy_env_raises_on_reset(self) -> None:
        from praxis_eval.envs.robomimic.runtime import _DummyRobomimicEnv

        env = _DummyRobomimicEnv(camera_names=["agentview"])
        with pytest.raises(NotImplementedError):
            env.reset()


class TestRobomimicEvalLaneWrapper:
    def test_task_description_uses_policy_instruction_after_retarget(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import gymnasium as gym
        from gymnasium import spaces

        from praxis_eval.envs.robomimic.runtime import RobomimicEvalLaneWrapper

        runtime_mod = importlib.import_module("praxis_eval.envs.robomimic.runtime")

        class _FakeEnv(gym.Env):
            image_size = 64
            camera_names = ["agentview"]
            state_ports = [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ]
            video_camera = "agentview"
            video_resolution = 64
            _max_episode_steps = 800
            robot = "Panda"
            _seed = 0

            def __init__(self, task_name: str) -> None:
                super().__init__()
                self.task_name = task_name
                self.closed = False
                self.observation_space = spaces.Dict({})
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(7,),
                    dtype=np.float32,
                )

            @property
            def task_description(self) -> str:
                return get_task_instruction(self.task_name)

            @property
            def task(self) -> str:
                return self.task_name

            def close(self) -> None:
                self.closed = True

            def reset(self, *, seed=None, options=None):
                _ = (seed, options)
                return {}, {}

        def _fake_build_robomimic_env_with_retries(**kwargs):
            return _FakeEnv(task_name=str(kwargs["task_name"]))

        monkeypatch.setattr(
            runtime_mod,
            "build_robomimic_env_with_retries",
            _fake_build_robomimic_env_with_retries,
        )

        env = _FakeEnv("Lift")
        wrapper = RobomimicEvalLaneWrapper(env, task_group="mt3", lane_idx=0)

        assert wrapper.task_description == "Lift the cube."
        wrapper.prepare_eval_job(task_id=1, episode_index=7)
        assert wrapper.task == "PickPlaceCan"
        assert (
            wrapper.task_description
            == "Pick up the can and place it in the target bin."
        )


class TestRobomimicProcessorStep:
    def test_flattens_robot_state(self) -> None:
        import torch
        from lerobot.utils.constants import OBS_PREFIX, OBS_STATE

        from praxis_eval.envs.robomimic.processor import RobomimicProcessorStep

        batch_size = 2
        obs = {
            f"{OBS_PREFIX}robot_state": {
                "robot0_eef_pos": torch.zeros(batch_size, 3),
                "robot0_eef_quat": torch.ones(batch_size, 4),
                "robot0_gripper_qpos": torch.full((batch_size, 2), 2.0),
            },
            "observation.images.agentview": torch.zeros(batch_size, 3, 64, 64),
        }
        proc = RobomimicProcessorStep()
        out = proc.observation(obs)

        assert OBS_STATE in out
        assert out[OBS_STATE].shape == (batch_size, 9)
        assert torch.all(out[OBS_STATE][:, :3] == 0.0)
        assert torch.all(out[OBS_STATE][:, 3:7] == 1.0)
        assert torch.all(out[OBS_STATE][:, 7:] == 2.0)
        assert "observation.state.robot0_eef_pos" in out
        assert "observation.images.agentview" in out
        assert f"{OBS_PREFIX}robot_state" not in out

    def test_env_preprocessor_uses_configured_state_order(self) -> None:
        from praxis_eval.processing import make_env_pre_post_processors

        proc, _ = make_env_pre_post_processors(
            RobomimicEnvConfig(
                state_ports=["robot0_joint_pos", "robot0_eef_pos"],
            ),
            policy_cfg=SimpleNamespace(),
        )
        step = proc.steps[0]
        assert step.state_ports == ("robot0_joint_pos", "robot0_eef_pos")


class TestRobomimicEnvWrapper:
    def test_step_reports_success_reward_and_gymnasium_done_tuple(self) -> None:
        from praxis_eval.envs.robomimic.env import RobomimicEnv

        class _FakeRawEnv:
            def __init__(self) -> None:
                self.step_actions: list[np.ndarray] = []

            def step(self, action):
                self.step_actions.append(np.asarray(action, dtype=np.float32))
                return _fake_raw_obs(), 0.0, False, {"raw": True}

            def _check_success(self) -> bool:
                return True

        raw_env = _FakeRawEnv()
        env = RobomimicEnv.__new__(RobomimicEnv)
        env.env = raw_env
        env.task_name = "Lift"
        env.state_ports = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        env.camera_names = ["agentview"]
        env.image_size = 64
        env.enable_render = True
        env._step_count = 0
        env._max_episode_steps = 800
        env._done = False
        env.action_space = types.SimpleNamespace(
            shape=(7,),
            dtype=np.float32,
            contains=lambda action: bool(
                np.asarray(action).shape == (7,)
                and np.all(np.asarray(action) >= -1.0)
                and np.all(np.asarray(action) <= 1.0)
            ),
        )

        assert env.task_description == "Lift the cube."

        obs, reward, terminated, truncated, info = RobomimicEnv.step(
            env,
            np.zeros((7,), dtype=np.float32),
        )

        assert "pixels" in obs
        assert reward == 1.0
        assert terminated is True
        assert truncated is False
        assert info["is_success"] is True
        assert info["final_info"]["task"] == "Lift"

    def test_step_reports_time_limit_as_truncated(self) -> None:
        from praxis_eval.envs.robomimic.env import RobomimicEnv

        class _FakeRawEnv:
            def step(self, action):
                _ = action
                return _fake_raw_obs(), 0.0, False, {}

            def _check_success(self) -> bool:
                return False

        env = RobomimicEnv.__new__(RobomimicEnv)
        env.env = _FakeRawEnv()
        env.task_name = "Lift"
        env.state_ports = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        env.camera_names = ["agentview"]
        env.image_size = 64
        env.enable_render = True
        env._step_count = 0
        env._max_episode_steps = 1
        env._done = False
        env.action_space = types.SimpleNamespace(
            shape=(7,),
            dtype=np.float32,
            contains=lambda action: bool(np.asarray(action).shape == (7,)),
        )

        _obs, reward, terminated, truncated, info = RobomimicEnv.step(
            env,
            np.zeros((7,), dtype=np.float32),
        )

        assert reward == 0.0
        assert terminated is False
        assert truncated is True
        assert info["final_info"]["truncated"] is True

    def test_step_validates_action_shape_and_bounds(self) -> None:
        from praxis_eval.envs.robomimic.env import RobomimicEnv

        env = RobomimicEnv.__new__(RobomimicEnv)
        env.action_space = types.SimpleNamespace(
            shape=(7,),
            dtype=np.float32,
            contains=lambda action: bool(
                np.asarray(action).shape == (7,)
                and np.all(np.asarray(action) >= -1.0)
                and np.all(np.asarray(action) <= 1.0)
            ),
        )

        with pytest.raises(ValueError, match="Expected RoboMimic action shape"):
            RobomimicEnv.step(env, np.zeros((8,), dtype=np.float32))
        with pytest.raises(ValueError, match="within the action space bounds"):
            RobomimicEnv.step(env, np.array([2.0, *([0.0] * 6)], dtype=np.float32))

    def test_create_env_uses_robosuite_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        from praxis_eval.envs.robomimic.env import RobomimicEnv

        seen: dict[str, Any] = {}

        class _FakeRawEnv:
            action_spec = (
                -np.ones((7,), dtype=np.float32),
                np.ones((7,), dtype=np.float32),
            )

            def __init__(self) -> None:
                self.hard_reset = True
                self.seed_calls: list[int] = []

            def _get_observations(self, force_update=True):
                _ = force_update
                return _fake_raw_obs()

            def reset(self):
                return _fake_raw_obs()

            def seed(self, seed):
                self.seed_calls.append(int(seed))

            def close(self):
                return None

        def _fake_make(**kwargs):
            seen.update(kwargs)
            return _FakeRawEnv()

        fake_robosuite = types.ModuleType("robosuite")
        fake_robosuite.make = _fake_make
        fake_controllers = types.ModuleType("robosuite.controllers")
        fake_controllers.load_composite_controller_config = lambda controller, robot: {
            "controller": controller,
            "robot": robot,
            "body_parts": {"right": {}},
        }
        monkeypatch.setitem(sys.modules, "robosuite", fake_robosuite)
        monkeypatch.setitem(sys.modules, "robosuite.controllers", fake_controllers)

        env = RobomimicEnv(
            task_name="Lift",
            image_size=64,
            seed=5,
            camera_names=["agentview"],
            max_episode_steps=50,
            enable_render=True,
        )
        env.close()

        assert seen["env_name"] == "Lift"
        assert seen["robots"] == "Panda"
        assert seen["camera_names"] == ["agentview"]
        assert seen["camera_widths"] == 64
        assert seen["camera_heights"] == 64
        assert seen["has_renderer"] is False
        assert seen["has_offscreen_renderer"] is True
        assert seen["use_camera_obs"] is True
        assert seen["control_freq"] == 20
        assert seen["reward_shaping"] is False
        assert seen["lite_physics"] is False
        assert seen["controller_configs"]["body_parts"]["right"] == {
            "input_type": "delta",
            "input_ref_frame": "world",
        }
        assert seen["seed"] == 5
