# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Managed cache paths used by praxis-eval runtime setup."""

from __future__ import annotations

import os
from pathlib import Path


def managed_asset_dir(env_name: str) -> Path:
    """Return the default ManiSkill asset directory for an eval family."""
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return (
        cache_home / "praxis_eval" / "assets" / env_name / "maniskill_assets"
    ).resolve()
