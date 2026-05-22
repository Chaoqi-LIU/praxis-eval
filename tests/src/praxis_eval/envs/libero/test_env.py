"""Tests for the Praxis LIBERO env adapter."""

from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from types import SimpleNamespace

import numpy as np
import pytest


def test_libero_env_adapts_praxis_robosuite_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_libero_pkg = types.ModuleType("libero")
    fake_libero_libero = types.ModuleType("libero.libero")
    fake_libero_envs = types.ModuleType("libero.libero.envs")
    fake_libero_pkg.__path__ = []
    fake_libero_libero.__path__ = []
    fake_libero_pkg.__spec__ = ModuleSpec("libero", loader=None, is_package=True)
    fake_libero_libero.__spec__ = ModuleSpec(
        "libero.libero",
        loader=None,
        origin="/tmp/libero/libero/__init__.py",
        is_package=True,
    )
    fake_libero_envs.__spec__ = ModuleSpec("libero.libero.envs", loader=None)
    fake_libero_pkg.libero = fake_libero_libero
    fake_libero_libero.envs = fake_libero_envs
    fake_libero_libero.benchmark = SimpleNamespace()
    fake_libero_libero.get_libero_path = lambda key: f"/tmp/libero/{key}"
    fake_libero_envs.OffScreenRenderEnv = object
    monkeypatch.setitem(sys.modules, "libero", fake_libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", fake_libero_libero)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", fake_libero_envs)

    import lerobot.envs.libero as lerobot_libero

    from praxis_eval.envs.libero.env import LiberoEnv

    created: list[str] = []

    class _FakeController:
        ref_ori_mat = np.eye(3)
        use_delta = False

    class _FakeBackend:
        def __init__(self, **kwargs) -> None:
            created.append(str(kwargs["bddl_file_name"]))
            self.env = SimpleNamespace(_get_observations=self._obs)
            self.robots = [
                SimpleNamespace(part_controllers={"right": _FakeController()})
            ]

        def _obs(self):
            return self._raw_obs()

        def _raw_obs(self):
            return {
                "agentview_image": np.zeros((4, 5, 3), dtype=np.uint8),
                "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": np.array([0.1, 0.2]),
                "robot0_gripper_qvel": np.array([0.3, 0.4]),
                "robot0_joint_pos": np.arange(7),
                "robot0_joint_vel": np.arange(7) + 10,
            }

        def seed(self, seed) -> None:
            self.seed_value = seed

        def reset(self):
            return self._raw_obs()

        def step(self, action):
            _ = action
            return self._raw_obs(), 0.0, False, {}

        def check_success(self) -> bool:
            return False

        def close(self) -> None:
            pass

    monkeypatch.setattr(lerobot_libero, "OffScreenRenderEnv", _FakeBackend)
    monkeypatch.setattr(
        lerobot_libero, "get_libero_path", lambda key: f"/tmp/libero/{key}"
    )

    task = SimpleNamespace(
        name="fake task",
        language="do fake thing",
        problem_folder="folder",
        bddl_file="task.bddl",
    )
    suite = SimpleNamespace(get_task=lambda task_id: task)
    env = LiberoEnv(
        task_suite=suite,
        task_id=0,
        task_suite_name="libero_10",
        camera_name=["agentview_image"],
        obs_type="pixels_agent_pos",
        init_states=False,
        observation_width=5,
        observation_height=4,
        num_steps_wait=0,
    )
    assert created == ["/tmp/libero/bddl_files/folder/task.bddl"]

    obs, info = env.reset(seed=123)
    assert info == {"is_success": False}
    assert env.task_description == "do fake thing"
    assert obs["pixels"]["image"].shape == (4, 5, 3)
    np.testing.assert_array_equal(
        obs["robot_state"]["eef"]["mat"],
        np.eye(3),
    )
    assert env._env.robots[0].part_controllers["right"].use_delta is True
