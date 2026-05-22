"""Unit tests for RoboCasa365 dataset selector helpers."""

from __future__ import annotations

import sys
import types

import pytest

from praxis_eval.envs.robocasa import datasets as ds


def test_expand_custom_task_selector_group() -> None:
    assert ds.expand_custom_task_selector(group="mt5") == [
        "CloseToasterOvenDoor",
        "OpenDrawer",
        "PickPlaceDrawerToCounter",
        "TurnOnElectricKettle",
        "SlideDishwasherRack",
    ]


def test_expand_custom_task_selector_task_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ds,
        "_task_sets",
        lambda: {"atomic_seen": ("OpenDrawer", "CloseDrawer")},
    )
    assert ds.expand_custom_task_selector(task_set="atomic_seen") == [
        "OpenDrawer",
        "CloseDrawer",
    ]


def test_pool_to_split_source() -> None:
    assert ds.pool_to_split_source("pretrain-human") == ("pretrain", "human")
    assert ds.pool_to_split_source("pretrain-mimicgen") == ("pretrain", "mg")
    assert ds.pool_to_split_source("target-human") == ("target", "human")


def test_resolve_pool_entries_uses_pool_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ds,
        "expand_custom_task_selector",
        lambda **_: ["OpenDrawer", "CloseDrawer"],
    )
    calls: list[tuple[str, str, str, float]] = []
    fake_robocasa = types.ModuleType("robocasa")
    fake_utils = types.ModuleType("robocasa.utils")
    fake_registry_utils = types.ModuleType("robocasa.utils.dataset_registry_utils")
    fake_robocasa.utils = fake_utils
    fake_utils.dataset_registry_utils = fake_registry_utils
    fake_registry_utils.get_ds_meta = lambda *, task, split, source, demo_fraction: (
        calls.append((task, split, source, demo_fraction))
        or {
            "task": task,
            "split": split,
            "source": source,
            "path": f"/tmp/{task}/lerobot",
        }
    )
    monkeypatch.setitem(sys.modules, "robocasa", fake_robocasa)
    monkeypatch.setitem(sys.modules, "robocasa.utils", fake_utils)
    monkeypatch.setitem(
        sys.modules,
        "robocasa.utils.dataset_registry_utils",
        fake_registry_utils,
    )

    monkeypatch.setattr(
        "robocasa.utils.dataset_registry_utils.get_ds_meta",
        fake_registry_utils.get_ds_meta,
    )

    entries = ds.resolve_pool_entries(group="mt5", pool="target-human")

    assert entries == [
        {
            "task": "OpenDrawer",
            "split": "target",
            "source": "human",
            "path": "/tmp/OpenDrawer/lerobot",
        },
        {
            "task": "CloseDrawer",
            "split": "target",
            "source": "human",
            "path": "/tmp/CloseDrawer/lerobot",
        },
    ]
    assert calls == [
        ("OpenDrawer", "target", "human", 1.0),
        ("CloseDrawer", "target", "human", 1.0),
    ]


def test_resolve_dataset_soup_entries_uses_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ds,
        "_dataset_soups",
        lambda: {
            "target_atomic_seen": (
                {
                    "task": "OpenDrawer",
                    "split": "target",
                    "source": "human",
                    "path": "/tmp/open/lerobot",
                },
            )
        },
    )

    entries = ds.resolve_dataset_soup_entries("target_atomic_seen")
    assert entries == [
        {
            "task": "OpenDrawer",
            "split": "target",
            "source": "human",
            "path": "/tmp/open/lerobot",
        }
    ]
