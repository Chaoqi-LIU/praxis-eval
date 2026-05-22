# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility patches for MetaWorld under current Gymnasium."""

from __future__ import annotations

import importlib


def patch_metaworld_env() -> None:
    """Patch MetaWorld render metadata for Gymnasium 1.1+ MuJoCo assertions."""
    try:
        sawyer_mod = importlib.import_module("metaworld.sawyer_xyz_env")
    except ModuleNotFoundError:
        return

    sawyer_env = getattr(sawyer_mod, "SawyerXYZEnv", None)
    if sawyer_env is None or getattr(
        sawyer_env,
        "_praxis_render_modes_compat",
        False,
    ):
        return

    metadata = dict(getattr(sawyer_env, "metadata", {}) or {})
    metadata["render_modes"] = ["human", "rgb_array", "depth_array", "rgbd_tuple"]
    sawyer_env.metadata = metadata
    sawyer_env._praxis_render_modes_compat = True
