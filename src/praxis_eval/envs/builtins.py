# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in environment driver registration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from praxis_eval.contracts import EnvContract
from praxis_eval.types import EvalConfig, EvalResult, Policy


@dataclass(frozen=True)
class BuiltinEnvDriver:
    """Driver for a built-in benchmark family."""

    contract: EnvContract

    def evaluate(self, *, policy: Policy, config: EvalConfig) -> EvalResult:
        from omegaconf import OmegaConf

        from praxis_eval.envs.eval_registry import (
            EvalDriverContext,
            run_env_runtime_driver,
        )
        from praxis_eval.envs.factory import build_env_config
        from praxis_eval.evaluation.artifacts import (
            resolve_eval_artifact_paths,
            write_eval_results_json,
        )
        from praxis_eval.processing import (
            make_env_pre_post_processors,
            make_policy_pre_post_processors,
        )

        env_cfg = {**dict(config.env_kwargs), "type": self.contract.env_type}
        if config.task is not None:
            env_cfg["task"] = config.task
        if config.task_ids is not None:
            env_cfg["task_ids"] = list(config.task_ids)

        eval_output_dir, eval_media_dir = resolve_eval_artifact_paths(
            config.output_dir,
        )
        policy_preprocessor, policy_postprocessor = make_policy_pre_post_processors()
        env_cfg_obj = build_env_config(env_cfg)
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg_obj,
            policy_cfg=None,
        )
        remote_address = getattr(policy, "address", None)
        eval_mode = "remote_grpc" if remote_address else "local_inproc"
        process_start = monotonic()
        runtime_hooks = config.runtime_hooks

        def phase_heartbeat(label: str) -> None:
            if runtime_hooks.phase_heartbeat is not None:
                runtime_hooks.phase_heartbeat(label)
            else:
                _heartbeat("PHASE", label, process_start)

        def progress_heartbeat(label: str) -> None:
            if runtime_hooks.progress_heartbeat is not None:
                runtime_hooks.progress_heartbeat(label)
            else:
                _heartbeat("HEARTBEAT", label, process_start)

        eval_start = monotonic()
        results = run_env_runtime_driver(
            OmegaConf.create(env_cfg),
            context=EvalDriverContext(
                cfg=OmegaConf.create(
                    {
                        "env": env_cfg,
                        "eval": {
                            "num_eval_per_task": config.num_eval_per_task,
                            "num_parallel_env": config.num_parallel_env,
                            "record_episodes_per_task": (
                                config.record_episodes_per_task
                            ),
                        },
                    }
                ),
                seed=int(config.seed),
                eval_mode=eval_mode,
                eval_output_dir=eval_output_dir,
                eval_media_dir=eval_media_dir,
                num_eval_per_task=int(config.num_eval_per_task),
                num_parallel_env=int(config.num_parallel_env),
                eval_record_episodes_per_task=int(config.record_episodes_per_task),
                eval_debug_verbose=bool(config.debug_verbose),
                eval_step_timeout_sec=config.step_timeout_sec,
                eval_rollout_failure_retries=int(config.rollout_failure_retries),
                eval_policy_kwargs=dict(config.policy_kwargs),
                eval_device="cpu",
                server_address=str(remote_address) if remote_address else None,
                policy=policy,
                policy_preprocessor=policy_preprocessor,
                policy_postprocessor=policy_postprocessor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                phase_heartbeat=phase_heartbeat,
                progress_heartbeat=progress_heartbeat,
                action_spec=self.contract.action,
            ),
        )
        results_path = write_eval_results_json(
            results=results,
            output_dir=eval_output_dir,
            metadata={
                **dict(config.metadata),
                "seed": int(config.seed),
                "mode": eval_mode,
                "num_eval_per_task": int(config.num_eval_per_task),
                "num_parallel_env": int(config.num_parallel_env),
                "record_episodes_per_task": int(config.record_episodes_per_task),
                "rollout_failure_retries": int(config.rollout_failure_retries),
                "duration_s": monotonic() - eval_start,
            },
        )
        return EvalResult(
            overall=results.get("overall", {}),
            per_task=results.get("per_task", {}),
            per_group=results.get("per_group", {}),
            artifacts={
                **dict(results.get("artifacts", {})),
                "results_path": str(results_path),
                "output_dir": str(Path(eval_output_dir)),
                "media_dir": str(Path(eval_media_dir)),
            },
            metadata={
                "env_type": self.contract.env_type,
                "mode": eval_mode,
            },
        )


def register_builtin_contract_drivers() -> None:
    """Register built-in env-family drivers."""
    from praxis_eval.envs.libero import CONTRACT as LIBERO_CONTRACT
    from praxis_eval.envs.metaworld import CONTRACT as METAWORLD_CONTRACT
    from praxis_eval.envs.mshab import CONTRACT as MSHAB_CONTRACT
    from praxis_eval.envs.robocasa import CONTRACT as ROBOCASA_CONTRACT
    from praxis_eval.envs.robomimic import CONTRACT as ROBOMIMIC_CONTRACT
    from praxis_eval.envs.simpler import CONTRACT as SIMPLER_CONTRACT
    from praxis_eval.registry import register_driver

    for contract in (
        LIBERO_CONTRACT,
        METAWORLD_CONTRACT,
        MSHAB_CONTRACT,
        ROBOCASA_CONTRACT,
        ROBOMIMIC_CONTRACT,
        SIMPLER_CONTRACT,
    ):
        register_driver(
            contract.env_type,
            BuiltinEnvDriver(contract),
            replace=False,
        )


def _heartbeat(kind: str, label: str, process_start: float) -> None:
    elapsed = monotonic() - process_start
    print(f"{kind} label={label} mono_elapsed={elapsed:.3f}s")
