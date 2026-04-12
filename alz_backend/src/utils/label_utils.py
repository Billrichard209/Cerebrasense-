"""Label handling helpers with explicit safeguards against silent remapping."""

from __future__ import annotations


def assert_explicit_label_mapping(source_labels: set[str], target_labels: set[str]) -> None:
    """Reject implicit label alignment when schemas do not already match exactly."""

    if source_labels != target_labels:
        raise ValueError(
            "Label schemas differ. Add an explicit, documented mapping instead of silently remapping labels."
        )
