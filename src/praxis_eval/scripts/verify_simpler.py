# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Verify that SimplerEnv can roll out to completion with random actions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from praxis_eval.managed_paths import managed_asset_dir
from praxis_eval.scripts._verify_common import (
    env_or_default_path,
    load_json,
    print_summary,
    resolve_env_python_bin,
    resolve_output_dir,
    run_command,
    validate_completed_episodes,
)

_SIMPLER_ENV_IDS = {
    "widowx_spoon_on_towel": "PutSpoonOnTableClothInScene-v1",
    "widowx_carrot_on_plate": "PutCarrotOnPlateInScene-v1",
    "widowx_stack_cube": "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "widowx_put_eggplant_in_basket": "PutEggplantInBasketScene-v1",
}


def default_ms_asset_dir() -> Path:
    return env_or_default_path(
        "MS_ASSET_DIR",
        managed_asset_dir("simpler"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="widowx_carrot_on_plate")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--env-name", default="simpler-praxis")
    parser.add_argument("--env-python-bin", default=None)
    parser.add_argument("--shader", default="default")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ms-asset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_python_bin = resolve_env_python_bin(args.env_python_bin, str(args.env_name))
    output_dir = resolve_output_dir(args.output_dir, "simpler")
    output_dir.mkdir(parents=True, exist_ok=True)

    task = str(args.task)
    env_id = str(args.env_id or _SIMPLER_ENV_IDS.get(task, task))
    ms_asset_dir = (
        Path(args.ms_asset_dir).expanduser().resolve()
        if args.ms_asset_dir is not None
        else default_ms_asset_dir()
    )
    results_path = output_dir / "results.json"
    record_dir = output_dir / "artifacts"
    env = os.environ.copy()
    env["MS_ASSET_DIR"] = str(ms_asset_dir)
    command = [
        env_python_bin,
        "-u",
        "-m",
        "simpler_env.real2sim_eval_maniskill3",
        "--env-id",
        env_id,
        "--shader",
        str(args.shader),
        "--num-envs",
        str(int(args.num_envs)),
        "--num-episodes",
        str(int(args.num_episodes)),
        "--seed",
        str(int(args.seed)),
        "--record-dir",
        record_dir,
        "--metrics-output-path",
        results_path,
        "--no-save-video",
    ]
    run_command(command, env=env)
    payload = load_json(results_path)
    validate_completed_episodes(verify_name="verify_simpler", payload=payload)
    print_summary(
        verify_name="verify_simpler",
        results_path=results_path,
        payload=payload,
    )


if __name__ == "__main__":
    main()
