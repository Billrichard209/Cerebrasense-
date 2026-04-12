"""Validation helpers for dataset separation and safety-sensitive backend rules."""

from __future__ import annotations


def assert_distinct_dataset_sources(source_a: str, source_b: str) -> None:
    """Ensure two dataset roots are not treated as the same source by mistake."""

    if source_a == source_b:
        raise ValueError("OASIS and Kaggle sources must remain distinct unless an explicit merge plan exists.")


def assert_primary_dataset(dataset_name: str) -> None:
    """Ensure the configured primary dataset stays aligned with project rules."""

    if dataset_name != "oasis":
        raise ValueError("The backend core must stay OASIS-first unless explicitly re-scoped.")
