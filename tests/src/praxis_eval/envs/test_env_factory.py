"""Tests for env config normalization and registry wiring."""

from __future__ import annotations

import importlib
import subprocess
import sys
import types

import pytest
from omegaconf import OmegaConf

from praxis_eval.envs.eval_pool import EvalLaneJob
from praxis_eval.envs.factory import (
    available_async_env_types,
    available_env_types,
    available_eval_pool_env_types,
    build_env_config,
    infer_eval_env_target,
    make_env,
    register_async_env_builder,
    register_env_builder,
)
from praxis_eval.envs.libero.eval import make_libero_eval_pool


class TestEnvFactory:
    def test_factory_import_does_not_import_heavy_libero_stack(self):
        script = """
import sys
import praxis_eval.envs.factory  # noqa: F401
forbidden = [
    "praxis_eval.envs.libero.env",
    "lerobot.envs.libero",
    "libero.libero.envs",
    "robosuite",
]
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit(f"unexpected heavy imports: {loaded}")
"""
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            text=True,
            capture_output=True,
        )

    def test_libero_runtime_import_does_not_import_heavy_libero_stack(self):
        script = """
import sys
import praxis_eval.envs.libero.runtime  # noqa: F401
forbidden = [
    "praxis_eval.envs.libero.env",
    "lerobot.envs.libero",
    "libero.libero.envs",
    "robosuite",
]
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit(f"unexpected heavy imports: {loaded}")
"""
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            text=True,
            capture_output=True,
        )

    def test_available_env_types_contains_defaults(self):
        names = available_env_types()
        assert "libero" in names
        assert "aloha" in names

    def test_available_async_env_types_contains_default_libero(self):
        assert "libero" in available_async_env_types()

    def test_default_family_registrations_are_family_owned(self):
        factory_mod = importlib.import_module("praxis_eval.envs.factory")
        libero_registration = importlib.import_module(
            "praxis_eval.envs.libero.registration"
        )
        metaworld_registration = importlib.import_module(
            "praxis_eval.envs.metaworld.registration"
        )
        assert "mshab" in available_env_types()
        assert "metaworld" in available_env_types()
        assert "robocasa" in available_env_types()
        assert "robomimic" in available_env_types()
        assert "simpler" in available_env_types()
        assert "libero" in available_eval_pool_env_types()
        assert "metaworld" in available_eval_pool_env_types()
        assert "robocasa" in available_eval_pool_env_types()
        assert "robomimic" in available_eval_pool_env_types()
        assert (
            factory_mod._ASYNC_ENV_BUILDER_REGISTRY["libero"]
            is libero_registration.build_libero_async_env
        )
        assert (
            factory_mod._ENV_BUILDER_REGISTRY["libero"]
            is libero_registration.build_libero_env
        )
        assert (
            factory_mod._ENV_BUILDER_REGISTRY["metaworld"]
            is metaworld_registration.build_metaworld_env
        )
        assert (
            factory_mod._EVAL_POOL_BUILDER_REGISTRY["robocasa"]
            == "praxis_eval.envs.robocasa.eval:build_robocasa_eval_pool"
        )
        assert (
            factory_mod._EVAL_POOL_BUILDER_REGISTRY["metaworld"]
            == "praxis_eval.envs.metaworld.eval:build_metaworld_eval_pool"
        )
        assert (
            factory_mod._EVAL_POOL_BUILDER_REGISTRY["robomimic"]
            == "praxis_eval.envs.robomimic.eval:build_robomimic_eval_pool"
        )

    def test_infer_eval_env_target_uses_registered_non_prefix_inferers(self):
        assert infer_eval_env_target("bridge_single_view") == ("simpler", "bridge")

    def test_infer_eval_env_target_matches_mshab_clean_training_subset(self):
        assert infer_eval_env_target("mshab_settable") == ("mshab", "pick,place")
        assert infer_eval_env_target("mshab_settable_clean") == (
            "mshab",
            "pick,place",
        )

    def test_infer_eval_env_target_keeps_explicit_full_mshab_names_full_suite(self):
        assert infer_eval_env_target("mshab_settable_full") == ("mshab", "set_table")
        assert infer_eval_env_target("mshab_settable_w_depth") == (
            "mshab",
            "set_table",
        )

    def test_build_env_config_from_canonical_libero_cfg(self):
        from praxis_eval.envs.libero.config import LiberoEnvConfig

        cfg = OmegaConf.create(
            {
                "type": "libero",
                "task": "libero_10",
                "episode_length": 600,
                "observation_height": 128,
                "observation_width": 128,
            }
        )
        env_cfg = build_env_config(cfg)
        assert isinstance(env_cfg, LiberoEnvConfig)
        assert env_cfg.type == "libero"
        assert env_cfg.task == "libero_10"
        assert env_cfg.episode_length == 600
        assert env_cfg.observation_height == 128
        assert env_cfg.observation_width == 128

    def test_libero_env_config_resolves_partial_camera_mapping(self):
        from praxis_eval.envs.libero.config import LiberoEnvConfig

        cfg = LiberoEnvConfig(
            camera_name="agentview_image,robot0_eye_in_hand_image",
            camera_name_mapping={"agentview_image": "front"},
        )

        assert cfg.camera_name_mapping == {
            "agentview_image": "front",
            "robot0_eye_in_hand_image": "image2",
        }
        assert cfg.features_map["pixels/agentview_image"] == "observation.images.front"
        assert (
            cfg.features_map["pixels/robot0_eye_in_hand_image"]
            == "observation.images.image2"
        )

    def test_unknown_env_type_raises(self):
        with pytest.raises(ValueError, match="Unknown env type"):
            build_env_config({"type": "does_not_exist"})

    def test_async_libero_dummy_env_uses_effective_gym_kwargs_dims(self, monkeypatch):
        libero_registration = importlib.import_module(
            "praxis_eval.envs.libero.registration"
        )
        captured: dict[str, object] = {}

        def _fake_create_libero_envs(**kwargs):
            captured.update(kwargs)
            return {"libero_10": {0: "stub_env"}}

        fake_env_module = types.ModuleType("praxis_eval.envs.libero.env")
        fake_env_module.create_libero_envs = _fake_create_libero_envs
        monkeypatch.setitem(
            sys.modules,
            "praxis_eval.envs.libero.env",
            fake_env_module,
        )
        monkeypatch.setattr(
            libero_registration,
            "_make_async_vector_env_cls",
            lambda _cfg_obj: "async_vector_cls",
        )
        monkeypatch.setattr(
            libero_registration,
            "parse_camera_names",
            lambda _s: ["agentview_image"],
        )
        cfg = OmegaConf.create(
            {
                "type": "libero",
                "task": "libero_10",
                "observation_height": 360,
                "observation_width": 360,
            }
        )
        out = make_env(cfg, n_envs=2, use_async_envs=True)
        assert out == {"libero_10": {0: "stub_env"}}
        # Dummy bootstrap mirrors explicit LiberoEnv config fields.
        assert captured["env_cls"] == "async_vector_cls"
        assert captured["gym_kwargs"]["observation_height"] == 360
        assert captured["gym_kwargs"]["observation_width"] == 360

    def test_sync_libero_make_env_uses_local_builder(self, monkeypatch):
        captured: dict[str, object] = {}

        def _fake_create_libero_envs(**kwargs):
            captured.update(kwargs)
            return {"libero_10": {0: "sync_env"}}

        fake_env_module = types.ModuleType("praxis_eval.envs.libero.env")
        fake_env_module.create_libero_envs = _fake_create_libero_envs
        monkeypatch.setitem(
            sys.modules,
            "praxis_eval.envs.libero.env",
            fake_env_module,
        )

        out = make_env({"type": "libero", "task": "libero_10"}, n_envs=2)

        assert out == {"libero_10": {0: "sync_env"}}
        assert captured["task"] == "libero_10"
        assert captured["n_envs"] == 2
        assert "AsyncVectorEnv" not in repr(captured["env_cls"])

    def test_make_env_uses_registered_async_builder(self, monkeypatch):
        factory_mod = importlib.import_module("praxis_eval.envs.factory")
        captured: dict[str, object] = {}

        def _custom_builder(cfg_obj, n_envs):
            captured["cfg_type"] = cfg_obj.type
            captured["n_envs"] = n_envs
            return {"custom_suite": {0: "custom_env"}}

        monkeypatch.setitem(
            factory_mod._ASYNC_ENV_BUILDER_REGISTRY, "libero", _custom_builder
        )

        cfg = OmegaConf.create({"type": "libero", "task": "libero_10"})
        out = make_env(cfg, n_envs=3, use_async_envs=True)
        assert out == {"custom_suite": {0: "custom_env"}}
        assert captured["cfg_type"] == "libero"
        assert captured["n_envs"] == 3

    def test_direct_env_make_env_fails_without_local_builder(self, monkeypatch):
        factory_mod = importlib.import_module("praxis_eval.envs.factory")
        monkeypatch.delitem(
            factory_mod._ASYNC_ENV_BUILDER_REGISTRY, "libero", raising=False
        )
        monkeypatch.delitem(factory_mod._ENV_BUILDER_REGISTRY, "libero", raising=False)

        cfg = OmegaConf.create({"type": "libero", "task": "libero_10"})
        with pytest.raises(ValueError, match="evaluator-owned"):
            make_env(cfg, n_envs=4, use_async_envs=True)

    def test_register_async_env_builder_normalizes_name(self, monkeypatch):
        factory_mod = importlib.import_module("praxis_eval.envs.factory")

        def _builder(cfg_obj, n_envs):  # pragma: no cover - registration only
            return {}

        register_async_env_builder("  LIBERO_CUSTOM  ", _builder)
        assert "libero_custom" in available_async_env_types()
        monkeypatch.delitem(
            factory_mod._ASYNC_ENV_BUILDER_REGISTRY, "libero_custom", raising=False
        )

    def test_register_env_builder_normalizes_name(self, monkeypatch):
        factory_mod = importlib.import_module("praxis_eval.envs.factory")

        def _builder(cfg_obj, n_envs, use_async_envs):  # pragma: no cover
            return {}

        register_env_builder("  LOCAL_CUSTOM  ", _builder)
        assert "local_custom" in factory_mod._ENV_BUILDER_REGISTRY
        monkeypatch.delitem(
            factory_mod._ENV_BUILDER_REGISTRY,
            "local_custom",
            raising=False,
        )

    def test_env_family_packages_lazy_export_configs(self):
        import praxis_eval.envs.mshab as mshab
        import praxis_eval.envs.robocasa as robocasa
        import praxis_eval.envs.robomimic as robomimic
        import praxis_eval.envs.simpler as simpler

        assert robocasa.RobocasaEnvConfig.__name__ == "RobocasaEnvConfig"
        assert robomimic.RobomimicEnvConfig.__name__ == "RobomimicEnvConfig"
        assert mshab.MshabEnvConfig.__name__ == "MshabEnvConfig"
        assert mshab.MshabTaskSpec.__name__ == "MshabTaskSpec"
        assert simpler.SimplerEnvConfig.__name__ == "SimplerEnvConfig"

    def test_make_libero_eval_pool_strips_task_ids_from_worker_ctor_kwargs(
        self, monkeypatch
    ):
        async_vec_mod = importlib.import_module("praxis_eval.envs.async_vector_env")
        libero_eval_mod = importlib.import_module("praxis_eval.envs.libero.eval")
        libero_runtime = importlib.import_module("praxis_eval.envs.libero.runtime")

        captured: dict[str, object] = {}
        suite_calls: list[str] = []

        fake_env_module = types.ModuleType("praxis_eval.envs.libero.env")
        fake_env_module.get_suite = lambda suite_name: (
            suite_calls.append(suite_name) or object()
        )
        monkeypatch.setitem(
            sys.modules,
            "praxis_eval.envs.libero.env",
            fake_env_module,
        )

        monkeypatch.setattr(
            libero_runtime,
            "make_dummy_libero_env_fn",
            lambda **kwargs: lambda: object(),
        )

        def _fake_make_libero_env_fn(**kwargs):
            captured.setdefault("env_kwargs", []).append(kwargs)
            return lambda: {
                "task_id": kwargs["task_id"],
                "episode_index": kwargs["episode_index"],
                "reset_stride": kwargs["reset_stride"],
                "gym_kwargs": kwargs["gym_kwargs"],
            }

        def _fake_construct_libero_eval_lane(
            env_fn, *, suite_name, lane_idx, debug_verbose
        ):
            return {
                "env": env_fn(),
                "suite_name": suite_name,
                "lane_idx": lane_idx,
                "debug_verbose": debug_verbose,
            }

        monkeypatch.setattr(
            libero_runtime, "make_libero_env_fn", _fake_make_libero_env_fn
        )
        monkeypatch.setattr(
            libero_runtime,
            "construct_libero_eval_lane",
            _fake_construct_libero_eval_lane,
        )

        class _FakeAsyncVectorEnv:
            def __init__(self, env_fns, *, dummy_env_fn):
                self.env_fns = env_fns
                self.dummy_env_fn = dummy_env_fn
                self.num_envs = len(env_fns)
                self.instances = [fn() for fn in env_fns]
                self.call_each_calls = []

            def call_each(self, name, *, args_list, kwargs_list):
                self.call_each_calls.append((name, args_list, kwargs_list))
                return tuple([None] * self.num_envs)

        monkeypatch.setattr(async_vec_mod, "AsyncVectorEnv", _FakeAsyncVectorEnv)
        monkeypatch.setattr(libero_eval_mod, "AsyncVectorEnv", _FakeAsyncVectorEnv)

        pool = make_libero_eval_pool(
            {
                "type": "libero",
                "task": "libero_10",
                "task_ids": [0],
            },
            n_envs=2,
        )

        assert pool.env_pool is None
        assert pool.num_envs == 2
        assert "env_kwargs" not in captured

        first_jobs = [
            EvalLaneJob("libero_10", 0, 0, 3),
            EvalLaneJob("libero_10", 1, 0, 7),
        ]
        pool.prepare_jobs(first_jobs)
        assert isinstance(pool.env_pool, _FakeAsyncVectorEnv)
        assert suite_calls == ["libero_10"]
        env_kwargs = captured["env_kwargs"]
        assert isinstance(env_kwargs, list)
        assert [kwargs["task_id"] for kwargs in env_kwargs] == [0, 1]
        assert [kwargs["episode_index"] for kwargs in env_kwargs] == [3, 7]
        assert [kwargs["reset_stride"] for kwargs in env_kwargs] == [2, 2]
        gym_kwargs = env_kwargs[0]["gym_kwargs"]
        assert isinstance(gym_kwargs, dict)
        assert "task_ids" not in gym_kwargs
        assert gym_kwargs["num_steps_wait"] == 20

        second_jobs = [
            EvalLaneJob("libero_10", 2, 1, 11),
            EvalLaneJob("libero_10", 3, 1, 13),
        ]
        first_pool = pool.env_pool
        pool.prepare_jobs(second_jobs)
        assert pool.env_pool is first_pool
        assert first_pool.call_each_calls == [
            (
                "prepare_eval_job",
                [(2, 11), (3, 13)],
                [{"task_group": "libero_10"}, {"task_group": "libero_10"}],
            )
        ]

    def test_libero_camera_parser_is_local(self):
        from praxis_eval.envs.libero.spec import parse_camera_names

        assert parse_camera_names("agentview_image, robot0_eye_in_hand_image") == [
            "agentview_image",
            "robot0_eye_in_hand_image",
        ]
        with pytest.raises(ValueError, match="empty"):
            parse_camera_names(" , ")
