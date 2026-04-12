"""Basic governance helpers for safety-sensitive decision-support workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True, frozen=True)
class ProjectPolicy:
    """Central safety policy for the backend core."""

    product_mode: str = "decision_support"
    diagnosis_allowed: bool = False
    silent_label_remap_allowed: bool = False
    implicit_dataset_merge_allowed: bool = False
    primary_dataset: str = "oasis"
    notes: str = "The backend assists review and research workflows. It is not diagnosis software."


DEFAULT_POLICY = ProjectPolicy()
PROJECT_POLICY = asdict(DEFAULT_POLICY)


def assert_decision_support_policy() -> None:
    """Raise an error if the project policy stops matching the required posture."""

    if DEFAULT_POLICY.diagnosis_allowed:
        raise ValueError("This backend must remain decision-support software, not diagnosis software.")


def get_policy_snapshot() -> dict[str, object]:
    """Return the active project policy as a JSON-safe dictionary."""

    return asdict(DEFAULT_POLICY)
