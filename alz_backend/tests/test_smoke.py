"""Smoke tests for central settings, safety rules, and label safeguards."""

from __future__ import annotations

import pytest

from src.configs.runtime import AppSettings
from src.data.inventory import build_dataset_inventory_snapshot
from src.data.registry import build_dataset_registry_snapshot
from src.security.governance import PROJECT_POLICY, assert_decision_support_policy
from src.utils.label_utils import assert_explicit_label_mapping


def test_project_is_decision_support_only() -> None:
    """Ensure the project policy remains aligned with decision-support usage."""

    assert PROJECT_POLICY["product_mode"] == "decision_support"
    assert_decision_support_policy()


def test_label_mapping_must_be_explicit() -> None:
    """Ensure mismatched label schemas do not pass silently."""

    with pytest.raises(ValueError):
        assert_explicit_label_mapping({"cn", "ad"}, {"nondemented", "demented"})


def test_dataset_registry_keeps_sources_separate() -> None:
    """Ensure default central settings do not collapse OASIS and Kaggle into one source."""

    settings = AppSettings.from_env()
    registry = build_dataset_registry_snapshot(settings)

    assert registry["primary_dataset"] == "oasis"
    assert registry["datasets"]["oasis"]["source_root"] != registry["datasets"]["kaggle"]["source_root"]


def test_dataset_inventory_builds_for_registered_sources() -> None:
    """Ensure inventory generation works for the registered dataset roots."""

    inventory = build_dataset_inventory_snapshot(AppSettings.from_env())
    assert "oasis" in inventory["datasets"]
    assert "kaggle" in inventory["datasets"]
