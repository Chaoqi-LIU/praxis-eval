"""Unit tests for the in-process RoboCasa GR-1 adapter."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from praxis_eval.envs.factory import (
    available_env_types,
    available_eval_pool_env_types,
    build_env_config,
    list_tasks,
)
from praxis_eval.envs.robocasa_gr1.spec import (
    GR1_ACTION_DIM,
    GR1_ACTION_DIMS,
    GR1_LANGUAGE_KEY,
    GR1_STATE_DIMS,
    GR1_VIDEO_KEYS,
    flatten_gr1_action,
    unflatten_gr1_action,
)
from praxis_eval.envs.robocasa_gr1.tasks import (
    GR1_TASKS,
    expand_gr1_tasks,
    infer_robocasa_gr1_eval_target_from_dataset,
    resolve_gr1_task,
)


def _raw_observation() -> dict[str, object]:
    obs: dict[str, object] = {
        key: np.full((256, 256, 3), index, dtype=np.uint8)
        for index, key in enumerate(GR1_VIDEO_KEYS, start=1)
    }
    obs.update(
        {
            key: np.linspace(-2.1, 0.2, width, dtype=np.float32)
            for key, width in GR1_STATE_DIMS.items()
        }
    )
    obs[GR1_LANGUAGE_KEY] = "unlocked_waist: test task"
    return obs


class _FakeGr1Env:
    def __init__(self) -> None:
        self.last_action = None
        self.closed = False

    def reset(self, *, seed=None, options=None):
        return _raw_observation(), {"seed": seed, "options": options}

    def step(self, action):
        self.last_action = action
        return _raw_observation(), 0.0, False, False, {"success": False}

    def close(self):
        self.closed = True


class TestGr1Tasks:
    def test_official_task_groups(self) -> None:
        assert len(GR1_TASKS) == 24
        assert len(expand_gr1_tasks("articulated_6")) == 6
        assert len(expand_gr1_tasks("rearrangement_18")) == 18
        assert len(set(GR1_TASKS)) == 24

    def test_short_and_class_aliases(self) -> None:
        full = GR1_TASKS[0]
        class_name = full.split("/", 1)[1]
        short_name = class_name.removesuffix("_GR1ArmsAndWaistFourierHands_Env")
        assert resolve_gr1_task(full) == full
        assert resolve_gr1_task(class_name) == full
        assert resolve_gr1_task(short_name) == full

    def test_dataset_inference(self) -> None:
        assert infer_robocasa_gr1_eval_target_from_dataset(
            "robocasa_gr1_articulated_6"
        ) == ("robocasa_gr1", "articulated_6")
        assert infer_robocasa_gr1_eval_target_from_dataset("robocasa_mt5") is None


class TestGr1ActionSchema:
    def test_flatten_round_trip_uses_nvidia_order(self) -> None:
        streams = {
            key: np.full((width,), index, dtype=np.float32)
            for index, (key, width) in enumerate(GR1_ACTION_DIMS.items(), start=1)
        }
        flat = flatten_gr1_action(streams)
        assert flat.shape == (GR1_ACTION_DIM,)
        assert list(flat[:7]) == [1.0] * 7
        assert list(flat[7:14]) == [2.0] * 7
        round_tripped = unflatten_gr1_action(flat)
        for key in GR1_ACTION_DIMS:
            np.testing.assert_array_equal(round_tripped[key], streams[key])

    def test_unflatten_rejects_invalid_shape_and_nonfinite(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            unflatten_gr1_action(np.zeros(GR1_ACTION_DIM - 1, dtype=np.float32))
        action = np.zeros(GR1_ACTION_DIM, dtype=np.float32)
        action[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            unflatten_gr1_action(action)


class TestGr1Factory:
    def test_registered_as_in_process_family(self) -> None:
        assert "robocasa_gr1" in available_env_types()
        assert "robocasa_gr1" in available_eval_pool_env_types()
        cfg = build_env_config({"type": "robocasa_gr1", "task": "articulated_6"})
        assert cfg.type == "robocasa_gr1"
        assert cfg.max_episode_steps == 720
        assert len(list_tasks({"type": "robocasa_gr1", "task": "all"})) == 24

    def test_task_ids_filter_group(self) -> None:
        tasks = list_tasks(
            {
                "type": "robocasa_gr1",
                "task": "articulated_6",
                "task_ids": [1, 4],
            }
        )
        assert tasks == [(GR1_TASKS[1], 1), (GR1_TASKS[4], 4)]


class TestGr1Processor:
    def test_emits_official_groot_keys(self) -> None:
        from praxis_eval.envs.robocasa_gr1.processor import RobocasaGr1ProcessorStep

        raw: dict[str, object] = {"task": ["unlocked_waist: test"]}
        raw["observation.robot_state"] = {
            key: torch.zeros((1, width), dtype=torch.float32)
            for key, width in GR1_STATE_DIMS.items()
        }
        for key in GR1_VIDEO_KEYS:
            raw[f"observation.images.{key}"] = torch.zeros(
                (1, 3, 256, 256), dtype=torch.float32
            )

        processed = RobocasaGr1ProcessorStep().observation(raw)
        assert set(GR1_STATE_DIMS) <= set(processed)
        assert set(GR1_VIDEO_KEYS) <= set(processed)
        assert processed[GR1_LANGUAGE_KEY] == ["unlocked_waist: test"]


class TestGr1EnvAdapter:
    def test_reset_step_and_physical_unit_actions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for module_name in (
            "robocasa_gr1",
            "robocasa_gr1.utils",
            "robocasa_gr1.utils.gym_utils",
        ):
            package = types.ModuleType(module_name)
            package.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, module_name, package)
        fake_registration = types.ModuleType(
            "robocasa_gr1.utils.gym_utils.gymnasium_groot"
        )
        monkeypatch.setitem(
            sys.modules,
            "robocasa_gr1.utils.gym_utils.gymnasium_groot",
            fake_registration,
        )
        fake = _FakeGr1Env()
        monkeypatch.setattr("gymnasium.make", lambda *args, **kwargs: fake)

        from praxis_eval.envs.robocasa_gr1.env import RobocasaGr1Env

        env = RobocasaGr1Env(GR1_TASKS[0], max_episode_steps=2)
        obs, info = env.reset(seed=3)
        assert env.observation_space.contains(obs)
        assert env.task_description == "unlocked_waist: test task"
        assert info["seed"] == 3

        # Absolute arm joints legitimately exceed [-1, 1]. The adapter must not clip.
        action = np.linspace(-2.1, 0.5, GR1_ACTION_DIM, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert fake.last_action is not None
        np.testing.assert_array_equal(flatten_gr1_action(fake.last_action), action)

        _, _, terminated, truncated, info = env.step(action)
        assert not terminated
        assert truncated
        assert info["final_info"]["is_success"] is False
        env.close()
        assert fake.closed
