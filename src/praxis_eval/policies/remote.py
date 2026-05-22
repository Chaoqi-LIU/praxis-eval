# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Remote policy adapter backed by praxis-remote."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from praxis_eval.contracts import ActionSpec
from praxis_eval.policies.actions import normalize_batched_action
from praxis_eval.types import Observation


class RemotePolicy:
    """Policy adapter that calls a remote `praxis_remote.PolicyServer`."""

    def __init__(
        self,
        address: str,
        *,
        timeout: float | None = None,
    ) -> None:
        self.address = str(address)
        try:
            from praxis_remote import PolicyClient
        except ImportError as exc:  # pragma: no cover - user setup branch
            raise ImportError(
                "RemotePolicy requires praxis-remote. Install with "
                "`pip install 'praxis-eval[remote]==0.1.0'`."
            ) from exc

        host, port = self.address.rsplit(":", 1)
        self.client = PolicyClient(host=host, port=int(port), timeout=timeout)

    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        self.client.reset(episode_ids=episode_ids)

    def close(self) -> None:
        self.client.close()

    def act(
        self,
        observations: Sequence[Observation],
        *,
        action_spec: ActionSpec | None = None,
        policy_kwargs: Mapping[str, Any] | None = None,
        episode_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        if len(observations) < 1:
            raise ValueError("observations must be non-empty")
        action = self.client.predict_action(
            observations,
            policy_kwargs=policy_kwargs,
            episode_ids=episode_ids,
        )
        return normalize_batched_action(
            action,
            batch_size=len(observations),
            action_spec=action_spec,
        )
