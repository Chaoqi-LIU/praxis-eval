# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO-owned task enumeration and dataset-to-eval-target inference."""

from __future__ import annotations

from typing import Any

from praxis_eval.envs.libero.config import LiberoEnvConfig, ensure_libero_config
from praxis_eval.envs.libero.output import suppress_libero_output
from praxis_eval.envs.libero.spec import select_task_ids

_LIBERO_ALL_TASKS = "libero_spatial,libero_object,libero_goal,libero_10"


def infer_libero_eval_target_from_dataset(dataset_name: str) -> tuple[str, str] | None:
    """Return a LIBERO eval target for dataset names with LIBERO-specific meaning."""
    if dataset_name == "libero":
        return ("libero", _LIBERO_ALL_TASKS)
    return None


def list_libero_tasks(
    raw_cfg: dict[str, Any],
    cfg_obj: LiberoEnvConfig,
    debug_verbose: bool = False,
) -> list[tuple[str, int]]:
    """List `(suite_name, task_id)` pairs for a LIBERO env config."""
    ensure_libero_config()
    from libero.libero import benchmark

    gym_kwargs = dict(raw_cfg.get("gym_kwargs") or {})
    task_ids_filter = cfg_obj.task_ids
    if task_ids_filter is None:
        task_ids_filter = gym_kwargs.get("task_ids")
    suite_names = [s.strip() for s in str(cfg_obj.task).split(",") if s.strip()]

    tasks: list[tuple[str, int]] = []
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_name in suite_names:
        suite_ctor = benchmark_dict.get(suite_name)
        if suite_ctor is None:
            available = ", ".join(sorted(benchmark_dict.keys()))
            raise ValueError(
                f"Unknown LIBERO suite '{suite_name}'. Available: {available}"
            )
        with suppress_libero_output(not debug_verbose):
            suite = suite_ctor()
        total = len(suite.tasks)
        selected_ids = select_task_ids(total, task_ids_filter)
        tasks.extend((suite_name, int(task_id)) for task_id in selected_ids)
    return tasks
