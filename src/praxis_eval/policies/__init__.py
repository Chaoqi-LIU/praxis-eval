# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Policy adapters for local and remote evaluation."""

from praxis_eval.policies.local import LocalPolicy

__all__ = ["LocalPolicy", "RemotePolicy"]


def __getattr__(name: str):
    """Lazily expose optional remote policy support."""
    if name == "RemotePolicy":
        from praxis_eval.policies.remote import RemotePolicy

        return RemotePolicy
    raise AttributeError(name)
