"""Subject-safe split planning helpers for future OASIS-2 labeled work."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings
from src.utils.io_utils import ensure_directory

from .base_dataset import canonicalize_optional_string
from .oasis2_metadata import (
    build_oasis2_metadata_template,
    load_oasis2_metadata_template,
    resolve_oasis2_metadata_template_path,
)


@dataclass(slots=True)
class OASIS2SplitPolicySummary:
    """Status summary for the first subject-safe OASIS-2 split-plan preview."""

    generated_at: str
    metadata_path: str
    plan_csv_path: str
    subject_count: int
    bucket_count: int
    holdout_candidate_subject_count: int
    development_candidate_subject_count: int
    labeled_candidate_subject_count: int
    notes: list[str]
    recommendations: list[str]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        return asdict(self)


def _stable_bucket(value: str, bucket_count: int) -> int:
    """Map one stable string to a deterministic bucket index."""

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % bucket_count


def _future_role_hint(bucket_index: int) -> str:
    """Return a conservative planning-only role hint from a bucket index."""

    if bucket_index == 0:
        return "holdout_candidate"
    return "development_candidate"


def build_oasis2_subject_safe_split_plan(
    settings: AppSettings | None = None,
    *,
    metadata_path: Path | None = None,
    output_path: Path | None = None,
    bucket_count: int = 5,
) -> OASIS2SplitPolicySummary:
    """Build a deterministic subject-safe split-plan preview from the metadata template."""

    resolved_settings = settings or AppSettings.from_env()
    resolved_metadata_path = resolve_oasis2_metadata_template_path(resolved_settings, metadata_path=metadata_path)
    if not resolved_metadata_path.exists():
        build_oasis2_metadata_template(resolved_settings, output_path=resolved_metadata_path)
    metadata_frame = load_oasis2_metadata_template(resolved_settings, metadata_path=resolved_metadata_path)
    if bucket_count < 2:
        raise ValueError("bucket_count must be at least 2 for a meaningful subject-safe split preview.")

    working = metadata_frame.copy()
    working["subject_id"] = working["subject_id"].map(canonicalize_optional_string)
    working["split_group_hint"] = working["split_group_hint"].map(canonicalize_optional_string)
    working["group_key"] = working["split_group_hint"].fillna(working["subject_id"])
    working["has_candidate_label"] = working["diagnosis_label"].notna() & working["diagnosis_label_name"].notna()

    subject_rows: list[dict[str, Any]] = []
    for group_key, subject_frame in working.groupby("group_key", sort=True):
        subject_ids = sorted({value for value in subject_frame["subject_id"].dropna().astype(str).tolist() if value})
        bucket_index = _stable_bucket(str(group_key), bucket_count)
        role_hint = _future_role_hint(bucket_index)
        candidate_label_row_count = int(subject_frame["has_candidate_label"].sum())
        subject_rows.append(
            {
                "split_group_hint": group_key,
                "subject_ids": "|".join(subject_ids),
                "primary_subject_id": subject_ids[0] if subject_ids else None,
                "session_count": int(subject_frame["session_id"].nunique()),
                "visit_count": int(subject_frame["visit_number"].nunique()),
                "metadata_row_count": int(len(subject_frame)),
                "candidate_label_row_count": candidate_label_row_count,
                "subject_safe_bucket": bucket_index,
                "future_role_hint": role_hint,
            }
        )

    subject_frame = pd.DataFrame(subject_rows).sort_values(
        by=["subject_safe_bucket", "primary_subject_id", "split_group_hint"],
        kind="stable",
    )
    resolved_output_path = (
        output_path
        if output_path is not None
        else resolved_settings.data_root / "interim" / "oasis2_subject_safe_split_plan.csv"
    )
    ensure_directory(resolved_output_path.parent)
    summary_json_path = resolved_output_path.with_name("oasis2_subject_safe_split_plan_summary.json")
    summary_md_path = resolved_settings.outputs_root / "reports" / "onboarding" / "oasis2_subject_safe_split_plan.md"
    ensure_directory(summary_md_path.parent)
    subject_frame.to_csv(resolved_output_path, index=False)

    holdout_candidate_subject_count = int((subject_frame["future_role_hint"] == "holdout_candidate").sum())
    development_candidate_subject_count = int((subject_frame["future_role_hint"] == "development_candidate").sum())
    labeled_candidate_subject_count = int((subject_frame["candidate_label_row_count"] > 0).sum())

    summary = OASIS2SplitPolicySummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        metadata_path=str(resolved_metadata_path),
        plan_csv_path=str(resolved_output_path),
        subject_count=int(len(subject_frame)),
        bucket_count=bucket_count,
        holdout_candidate_subject_count=holdout_candidate_subject_count,
        development_candidate_subject_count=development_candidate_subject_count,
        labeled_candidate_subject_count=labeled_candidate_subject_count,
        notes=[
            "This is a planning-only subject-safe partition preview based on split_group_hint or subject_id.",
            "It does not create train/val/test manifests yet and does not override the need for label-aware split design.",
            "The stable bucket assignment is deterministic so future split work can stay reproducible.",
        ],
        recommendations=[
            "Keep split_group_hint aligned to the true patient-safe grouping key before any labeled OASIS-2 experiment.",
            "Do not turn this preview directly into model training splits until label coverage and class balance are reviewed.",
            "Once metadata is filled, use labeled_candidate_subject_count to decide whether supervised OASIS-2 evaluation is realistic.",
        ],
    )

    summary_payload = summary.to_payload()
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    lines = [
        "# OASIS-2 Subject-Safe Split Plan",
        "",
        f"- generated_at: {summary.generated_at}",
        f"- metadata_path: {summary.metadata_path}",
        f"- plan_csv_path: {summary.plan_csv_path}",
        f"- subject_count: {summary.subject_count}",
        f"- bucket_count: {summary.bucket_count}",
        f"- holdout_candidate_subject_count: {summary.holdout_candidate_subject_count}",
        f"- development_candidate_subject_count: {summary.development_candidate_subject_count}",
        f"- labeled_candidate_subject_count: {summary.labeled_candidate_subject_count}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.notes)
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in summary.recommendations)
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")
    return summary
