"""Supervised OASIS-2 split materialization and training-readiness checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory

from .base_dataset import canonicalize_optional_string, load_manifest_frame, parse_manifest_meta
from .oasis2_dataset import OASIS2_MANIFEST_COLUMNS
from .oasis2_metadata import resolve_oasis2_labeled_prep_manifest_path

OASIS2_SPLIT_PLAN_REQUIRED_COLUMNS = {
    "split_group_hint",
    "subject_ids",
    "primary_subject_id",
    "session_count",
    "visit_count",
    "metadata_row_count",
    "candidate_label_row_count",
    "subject_safe_bucket",
    "future_role_hint",
}
OASIS2_ALLOWED_BINARY_LABELS = {0, 1}


class OASIS2SupervisedError(ValueError):
    """Raised when supervised OASIS-2 preparation cannot proceed safely."""


@dataclass(slots=True, frozen=True)
class OASIS2SupervisedSplitConfig:
    """Configuration for subject-safe supervised OASIS-2 split materialization."""

    settings: AppSettings | None = None
    manifest_path: Path | None = None
    split_plan_path: Path | None = None
    reports_root: Path | None = None
    seed: int = 42
    split_seed: int | None = None
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass(slots=True)
class OASIS2SupervisedSplitArtifacts:
    """Saved split manifests and in-memory frames for supervised OASIS-2 work."""

    report_root: Path
    group_assignments_path: Path
    train_manifest_path: Path
    val_manifest_path: Path
    test_manifest_path: Path
    summary_json_path: Path
    summary_md_path: Path
    group_frame: pd.DataFrame
    assignments: pd.DataFrame
    merged_frame: pd.DataFrame
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    summary_payload: dict[str, Any]


@dataclass(slots=True, frozen=True)
class OASIS2TrainingReadinessCheck:
    """One supervised-training readiness finding."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OASIS2TrainingReadinessReport:
    """Serialized supervised-training readiness report."""

    generated_at: str
    labeled_manifest_path: str
    split_plan_path: str
    overall_status: str
    checks: list[OASIS2TrainingReadinessCheck]
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        return {
            "generated_at": self.generated_at,
            "labeled_manifest_path": self.labeled_manifest_path,
            "split_plan_path": self.split_plan_path,
            "overall_status": self.overall_status,
            "summary": _status_counts(self.checks),
            "dataset_summary": dict(self.dataset_summary),
            "checks": [asdict(check) for check in self.checks],
            "recommendations": list(self.recommendations),
            "notes": list(self.notes),
        }


def _status_counts(checks: list[OASIS2TrainingReadinessCheck]) -> dict[str, int]:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return counts


def _overall_status(checks: list[OASIS2TrainingReadinessCheck]) -> str:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def resolve_oasis2_subject_safe_split_plan_path(
    settings: AppSettings,
    *,
    split_plan_path: Path | None = None,
) -> Path:
    """Resolve the OASIS-2 subject-safe split-plan CSV path."""

    if split_plan_path is not None:
        return split_plan_path
    return settings.data_root / "interim" / "oasis2_subject_safe_split_plan.csv"


def load_oasis2_labeled_prep_manifest(
    settings: AppSettings | None = None,
    *,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Load the merged OASIS-2 labeled-prep manifest."""

    resolved_settings = settings or get_app_settings()
    resolved_path = resolve_oasis2_labeled_prep_manifest_path(resolved_settings, output_path=manifest_path)
    return load_manifest_frame(
        resolved_path,
        required_columns=OASIS2_MANIFEST_COLUMNS,
        default_dataset_type="3d_volumes",
    )


def load_oasis2_subject_safe_split_plan(
    settings: AppSettings | None = None,
    *,
    split_plan_path: Path | None = None,
) -> pd.DataFrame:
    """Load the OASIS-2 subject-safe split-plan CSV."""

    resolved_settings = settings or get_app_settings()
    resolved_path = resolve_oasis2_subject_safe_split_plan_path(
        resolved_settings,
        split_plan_path=split_plan_path,
    )
    frame = pd.read_csv(resolved_path)
    missing = OASIS2_SPLIT_PLAN_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise OASIS2SupervisedError(
            f"OASIS-2 subject-safe split plan is missing required columns: {sorted(missing)}"
        )
    if frame.empty:
        raise OASIS2SupervisedError("OASIS-2 subject-safe split plan is empty.")
    return frame


def _validate_split_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise OASIS2SupervisedError(
            "OASIS-2 split fractions must sum to 1.0, "
            f"got train={train_fraction}, val={val_fraction}, test={test_fraction}."
        )
    for name, value in (("train", train_fraction), ("val", val_fraction), ("test", test_fraction)):
        if value <= 0 or value >= 1:
            raise OASIS2SupervisedError(f"{name} fraction must be between 0 and 1, got {value}.")


def _normalize_binary_label(value: Any) -> int:
    if value is None or pd.isna(value):
        raise OASIS2SupervisedError("OASIS-2 labeled-prep manifest still contains missing labels.")
    normalized = int(float(value))
    if normalized not in OASIS2_ALLOWED_BINARY_LABELS:
        raise OASIS2SupervisedError(
            "OASIS-2 supervised training currently supports only binary labels {0, 1}. "
            f"Found {normalized!r}."
        )
    return normalized


def _resolve_split_group_hint(row: pd.Series) -> str:
    meta = parse_manifest_meta(row.get("meta"))
    metadata_payload = meta.get("oasis2_metadata")
    if isinstance(metadata_payload, dict):
        group_hint = canonicalize_optional_string(metadata_payload.get("split_group_hint"))
        if group_hint:
            return group_hint
    subject_id = canonicalize_optional_string(row.get("subject_id"))
    if subject_id:
        return subject_id
    raise OASIS2SupervisedError("OASIS-2 labeled-prep manifest row is missing both split_group_hint and subject_id.")


def _report_folder_name(cfg: OASIS2SupervisedSplitConfig) -> str:
    resolved_split_seed = cfg.seed if cfg.split_seed is None else cfg.split_seed
    return (
        f"oasis2_supervised_seed{cfg.seed}"
        f"_split{resolved_split_seed}"
        f"_train{int(round(cfg.train_fraction * 100))}"
        f"_val{int(round(cfg.val_fraction * 100))}"
        f"_test{int(round(cfg.test_fraction * 100))}"
    )


def _resolve_reports_root(cfg: OASIS2SupervisedSplitConfig, settings: AppSettings) -> Path:
    base_root = cfg.reports_root or (settings.outputs_root / "reports")
    return ensure_directory(Path(base_root) / _report_folder_name(cfg))


def _prepare_supervised_manifest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy().reset_index(drop=True)
    working["subject_id"] = working["subject_id"].map(canonicalize_optional_string)
    working["session_id"] = working["session_id"].map(canonicalize_optional_string)
    working["label_name"] = working["label_name"].map(canonicalize_optional_string)
    working["split_group_key"] = working.apply(_resolve_split_group_hint, axis=1)
    working["label"] = working["label"].map(_normalize_binary_label)

    if working["label_name"].isna().any():
        raise OASIS2SupervisedError(
            "OASIS-2 labeled-prep manifest still contains missing label_name values. "
            "Fill diagnosis_label_name before supervised training."
        )
    if working["subject_id"].isna().any():
        raise OASIS2SupervisedError("OASIS-2 labeled-prep manifest is missing subject_id values.")
    if working["session_id"].isna().any():
        raise OASIS2SupervisedError("OASIS-2 labeled-prep manifest is missing session_id values.")
    if working["session_id"].duplicated().any():
        duplicate_examples = working.loc[working["session_id"].duplicated(keep=False), "session_id"].drop_duplicates().tolist()
        raise OASIS2SupervisedError(
            "OASIS-2 supervised split materialization requires unique session_id values. "
            f"Examples: {duplicate_examples[:10]}"
        )
    return working


def _build_group_frame(
    manifest_frame: pd.DataFrame,
    split_plan_frame: pd.DataFrame,
) -> pd.DataFrame:
    plan = split_plan_frame.copy().reset_index(drop=True)
    plan["split_group_hint"] = plan["split_group_hint"].map(canonicalize_optional_string)
    if plan["split_group_hint"].isna().any():
        raise OASIS2SupervisedError("OASIS-2 split plan contains empty split_group_hint values.")
    if plan["split_group_hint"].duplicated().any():
        duplicates = plan.loc[plan["split_group_hint"].duplicated(keep=False), "split_group_hint"].drop_duplicates().tolist()
        raise OASIS2SupervisedError(
            "OASIS-2 split plan contains duplicate split_group_hint rows. "
            f"Examples: {duplicates[:10]}"
        )

    group_rows: list[dict[str, Any]] = []
    for split_group_key, group in manifest_frame.groupby("split_group_key", sort=True):
        sorted_subject_ids = sorted(
            {
                str(value)
                for value in group["subject_id"].dropna().astype(str).tolist()
                if str(value).strip()
            }
        )
        label_values = sorted({int(value) for value in group["label"].tolist()})
        group_rows.append(
            {
                "split_group_key": split_group_key,
                "primary_subject_id": sorted_subject_ids[0] if sorted_subject_ids else None,
                "subject_ids": "|".join(sorted_subject_ids),
                "subject_count": len(sorted_subject_ids),
                "session_count": int(len(group)),
                "visit_count": int(group["visit_number"].nunique()),
                "negative_session_count": int((group["label"] == 0).sum()),
                "positive_session_count": int((group["label"] == 1).sum()),
                "label_values": json.dumps(label_values),
                "group_binary_label": int((group["label"] == 1).any()),
                "mixed_label_group": bool(len(label_values) > 1),
            }
        )

    group_frame = pd.DataFrame(group_rows)
    merged = group_frame.merge(
        plan,
        left_on="split_group_key",
        right_on="split_group_hint",
        how="left",
        validate="one_to_one",
    )
    if merged["subject_safe_bucket"].isna().any():
        missing = merged.loc[merged["subject_safe_bucket"].isna(), "split_group_key"].drop_duplicates().tolist()
        raise OASIS2SupervisedError(
            "OASIS-2 split plan does not cover every labeled split_group_hint. "
            f"Examples: {missing[:10]}"
        )

    extra_plan_groups = sorted(set(plan["split_group_hint"].tolist()) - set(group_frame["split_group_key"].tolist()))
    if extra_plan_groups:
        raise OASIS2SupervisedError(
            "OASIS-2 split plan contains groups that are not present in the labeled-prep manifest. "
            f"Examples: {extra_plan_groups[:10]}"
        )
    return merged.sort_values("split_group_key").reset_index(drop=True)


def _assign_group_splits(
    group_frame: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> pd.DataFrame:
    _validate_split_fractions(train_fraction, val_fraction, test_fraction)
    if group_frame.empty:
        raise OASIS2SupervisedError("OASIS-2 supervised split materialization needs at least one split group.")

    class_counts = group_frame["group_binary_label"].value_counts().sort_index()
    if sorted(class_counts.index.tolist()) != [0, 1]:
        raise OASIS2SupervisedError(
            "OASIS-2 supervised training requires both binary classes {0, 1} at the split-group level."
        )

    try:
        train_groups, temp_groups = train_test_split(
            group_frame,
            test_size=(val_fraction + test_fraction),
            random_state=seed,
            stratify=group_frame["group_binary_label"],
        )
        temp_test_fraction = test_fraction / (val_fraction + test_fraction)
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=temp_test_fraction,
            random_state=seed,
            stratify=temp_groups["group_binary_label"],
        )
    except ValueError as error:
        raise OASIS2SupervisedError(
            "Could not create a safe stratified OASIS-2 supervised split. "
            "This usually means one class has too few patient groups for the requested fractions."
        ) from error

    assignments = pd.concat(
        [
            train_groups.assign(split="train"),
            val_groups.assign(split="val"),
            test_groups.assign(split="test"),
        ],
        ignore_index=True,
    )
    return assignments.sort_values("split_group_key").reset_index(drop=True)


def _apply_assignments(manifest_frame: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    merged = manifest_frame.merge(
        assignments[["split_group_key", "split", "group_binary_label", "mixed_label_group", "subject_safe_bucket", "future_role_hint"]],
        on="split_group_key",
        how="left",
        validate="many_to_one",
    )
    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), "split_group_key"].drop_duplicates().tolist()
        raise OASIS2SupervisedError(
            "OASIS-2 supervised assignments did not cover every manifest group. "
            f"Examples: {missing[:10]}"
        )
    return merged


def _save_split_reports(
    merged_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    cfg: OASIS2SupervisedSplitConfig,
    settings: AppSettings,
) -> OASIS2SupervisedSplitArtifacts:
    report_root = _resolve_reports_root(cfg, settings)
    group_assignments_path = report_root / "oasis2_supervised_group_assignments.csv"
    train_manifest_path = report_root / "oasis2_train_manifest.csv"
    val_manifest_path = report_root / "oasis2_val_manifest.csv"
    test_manifest_path = report_root / "oasis2_test_manifest.csv"
    summary_json_path = report_root / "oasis2_supervised_split_summary.json"
    summary_md_path = report_root / "oasis2_supervised_split_summary.md"

    train_frame = merged_frame.loc[merged_frame["split"] == "train"].copy().reset_index(drop=True)
    val_frame = merged_frame.loc[merged_frame["split"] == "val"].copy().reset_index(drop=True)
    test_frame = merged_frame.loc[merged_frame["split"] == "test"].copy().reset_index(drop=True)

    assignments.to_csv(group_assignments_path, index=False)
    train_frame.to_csv(train_manifest_path, index=False)
    val_frame.to_csv(val_manifest_path, index=False)
    test_frame.to_csv(test_manifest_path, index=False)

    subject_sets = {
        "train": set(train_frame["subject_id"].dropna().astype(str).tolist()),
        "val": set(val_frame["subject_id"].dropna().astype(str).tolist()),
        "test": set(test_frame["subject_id"].dropna().astype(str).tolist()),
    }
    group_sets = {
        "train": set(train_frame["split_group_key"].dropna().astype(str).tolist()),
        "val": set(val_frame["split_group_key"].dropna().astype(str).tolist()),
        "test": set(test_frame["split_group_key"].dropna().astype(str).tolist()),
    }
    mixed_label_group_count = int(assignments["mixed_label_group"].fillna(False).astype(bool).sum())
    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "oasis2",
        "task": "binary_structural_session_classification",
        "seed": cfg.seed,
        "split_seed": cfg.seed if cfg.split_seed is None else cfg.split_seed,
        "fractions": {
            "train": cfg.train_fraction,
            "val": cfg.val_fraction,
            "test": cfg.test_fraction,
        },
        "group_count": int(assignments["split_group_key"].nunique()),
        "subject_count": int(merged_frame["subject_id"].nunique()),
        "session_count": int(len(merged_frame)),
        "mixed_label_group_count": mixed_label_group_count,
        "group_counts": {
            split_name: int(frame["split_group_key"].nunique())
            for split_name, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame))
        },
        "subject_counts": {
            split_name: int(frame["subject_id"].nunique())
            for split_name, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame))
        },
        "row_counts": {
            split_name: int(len(frame))
            for split_name, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame))
        },
        "session_label_distribution_by_split": {
            split_name: {
                str(label): int(count)
                for label, count in frame["label"].value_counts().sort_index().to_dict().items()
            }
            for split_name, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame))
        },
        "group_label_distribution_by_split": {
            split_name: {
                str(label): int(count)
                for label, count in frame[["split_group_key", "group_binary_label"]]
                .drop_duplicates()["group_binary_label"]
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            }
            for split_name, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame))
        },
        "subject_overlap": {
            "train_val": sorted(subject_sets["train"].intersection(subject_sets["val"])),
            "train_test": sorted(subject_sets["train"].intersection(subject_sets["test"])),
            "val_test": sorted(subject_sets["val"].intersection(subject_sets["test"])),
        },
        "group_overlap": {
            "train_val": sorted(group_sets["train"].intersection(group_sets["val"])),
            "train_test": sorted(group_sets["train"].intersection(group_sets["test"])),
            "val_test": sorted(group_sets["val"].intersection(group_sets["test"])),
        },
        "artifacts": {
            "group_assignments_path": str(group_assignments_path),
            "train_manifest_path": str(train_manifest_path),
            "val_manifest_path": str(val_manifest_path),
            "test_manifest_path": str(test_manifest_path),
        },
        "notes": [
            "All sessions from the same split_group_hint stay in one split.",
            "Session labels remain session-level even when one patient group contains mixed longitudinal labels.",
            "Group-level stratification uses an ever-positive binary label so positive patient groups stay balanced.",
        ],
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    lines = [
        "# OASIS-2 Supervised Split Summary",
        "",
        f"- seed: {summary_payload['seed']}",
        f"- split_seed: {summary_payload['split_seed']}",
        f"- group_count: {summary_payload['group_count']}",
        f"- subject_count: {summary_payload['subject_count']}",
        f"- session_count: {summary_payload['session_count']}",
        f"- mixed_label_group_count: {summary_payload['mixed_label_group_count']}",
        "",
        "## Row Counts",
        "",
        f"- train: {summary_payload['row_counts']['train']}",
        f"- val: {summary_payload['row_counts']['val']}",
        f"- test: {summary_payload['row_counts']['test']}",
        "",
        "## Subject Counts",
        "",
        f"- train: {summary_payload['subject_counts']['train']}",
        f"- val: {summary_payload['subject_counts']['val']}",
        f"- test: {summary_payload['subject_counts']['test']}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in summary_payload["notes"])
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")

    return OASIS2SupervisedSplitArtifacts(
        report_root=report_root,
        group_assignments_path=group_assignments_path,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        test_manifest_path=test_manifest_path,
        summary_json_path=summary_json_path,
        summary_md_path=summary_md_path,
        group_frame=assignments.copy(),
        assignments=assignments,
        merged_frame=merged_frame,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        summary_payload=summary_payload,
    )


def build_oasis2_supervised_split_artifacts(
    cfg: OASIS2SupervisedSplitConfig | None = None,
    *,
    settings: AppSettings | None = None,
) -> OASIS2SupervisedSplitArtifacts:
    """Materialize subject-safe supervised OASIS-2 split manifests."""

    resolved_cfg = cfg or OASIS2SupervisedSplitConfig()
    resolved_settings = resolved_cfg.settings or settings or get_app_settings()
    manifest_frame = load_oasis2_labeled_prep_manifest(
        resolved_settings,
        manifest_path=resolved_cfg.manifest_path,
    )
    split_plan_frame = load_oasis2_subject_safe_split_plan(
        resolved_settings,
        split_plan_path=resolved_cfg.split_plan_path,
    )
    prepared_manifest = _prepare_supervised_manifest_frame(manifest_frame)
    group_frame = _build_group_frame(prepared_manifest, split_plan_frame)
    assignments = _assign_group_splits(
        group_frame,
        seed=resolved_cfg.seed if resolved_cfg.split_seed is None else resolved_cfg.split_seed,
        train_fraction=resolved_cfg.train_fraction,
        val_fraction=resolved_cfg.val_fraction,
        test_fraction=resolved_cfg.test_fraction,
    )
    merged_frame = _apply_assignments(prepared_manifest, assignments)
    return _save_split_reports(
        merged_frame,
        assignments,
        cfg=resolved_cfg,
        settings=resolved_settings,
    )


def build_oasis2_training_readiness_report(
    settings: AppSettings | None = None,
    *,
    manifest_path: Path | None = None,
    split_plan_path: Path | None = None,
    seed: int = 42,
    split_seed: int | None = None,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> OASIS2TrainingReadinessReport:
    """Inspect whether OASIS-2 is honestly ready for supervised training."""

    resolved_settings = settings or get_app_settings()
    resolved_manifest_path = resolve_oasis2_labeled_prep_manifest_path(
        resolved_settings,
        output_path=manifest_path,
    )
    resolved_split_plan_path = resolve_oasis2_subject_safe_split_plan_path(
        resolved_settings,
        split_plan_path=split_plan_path,
    )

    checks: list[OASIS2TrainingReadinessCheck] = []
    notes: list[str] = [
        "This readiness report is stricter than the onboarding checks because it guards actual supervised training.",
        "OASIS-2 training stays blocked until explicit labels, label names, and subject-safe splits are all valid.",
    ]
    recommendations: list[str] = []
    dataset_summary: dict[str, Any] = {
        "labeled_manifest_exists": resolved_manifest_path.exists(),
        "split_plan_exists": resolved_split_plan_path.exists(),
        "labeled_row_count": 0,
        "unlabeled_row_count": 0,
        "unique_subject_count": 0,
        "unique_session_count": 0,
        "split_group_count": 0,
        "mixed_label_group_count": 0,
    }

    if not resolved_manifest_path.exists():
        checks.append(
            OASIS2TrainingReadinessCheck(
                name="labeled_prep_manifest",
                status="fail",
                message="The OASIS-2 labeled-prep manifest does not exist yet.",
                details={"manifest_path": str(resolved_manifest_path)},
            )
        )
        recommendations.append(
            "Run the metadata adapter after filling the OASIS-2 metadata template so `oasis2_labeled_prep_manifest.csv` exists."
        )
    if not resolved_split_plan_path.exists():
        checks.append(
            OASIS2TrainingReadinessCheck(
                name="subject_safe_split_plan",
                status="fail",
                message="The OASIS-2 subject-safe split plan does not exist yet.",
                details={"split_plan_path": str(resolved_split_plan_path)},
            )
        )
        recommendations.append(
            "Build the OASIS-2 split plan from the metadata template before attempting supervised splits."
        )
    if checks:
        return OASIS2TrainingReadinessReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            labeled_manifest_path=str(resolved_manifest_path),
            split_plan_path=str(resolved_split_plan_path),
            overall_status=_overall_status(checks),
            checks=checks,
            dataset_summary=dataset_summary,
            recommendations=recommendations,
            notes=notes,
        )

    manifest_frame = load_oasis2_labeled_prep_manifest(
        resolved_settings,
        manifest_path=resolved_manifest_path,
    )
    split_plan_frame = load_oasis2_subject_safe_split_plan(
        resolved_settings,
        split_plan_path=resolved_split_plan_path,
    )
    dataset_summary.update(
        {
            "labeled_row_count": int(manifest_frame["label"].notna().sum()),
            "unlabeled_row_count": int(manifest_frame["label"].isna().sum()),
            "unique_subject_count": int(manifest_frame["subject_id"].nunique()),
            "unique_session_count": int(manifest_frame["session_id"].nunique()),
        }
    )
    checks.append(
        OASIS2TrainingReadinessCheck(
            name="labeled_prep_manifest",
            status="pass",
            message="The OASIS-2 labeled-prep manifest exists.",
            details={
                "manifest_path": str(resolved_manifest_path),
                "row_count": int(len(manifest_frame)),
            },
        )
    )
    checks.append(
        OASIS2TrainingReadinessCheck(
            name="subject_safe_split_plan",
            status="pass",
            message="The OASIS-2 subject-safe split plan exists.",
            details={
                "split_plan_path": str(resolved_split_plan_path),
                "row_count": int(len(split_plan_frame)),
            },
        )
    )

    unlabeled_row_count = int(manifest_frame["label"].isna().sum())
    missing_label_name_count = int(manifest_frame["label_name"].isna().sum())
    label_coverage_status = "pass" if unlabeled_row_count == 0 and missing_label_name_count == 0 else "fail"
    checks.append(
        OASIS2TrainingReadinessCheck(
            name="label_coverage",
            status=label_coverage_status,
            message=(
                "All OASIS-2 rows have explicit labels and label names."
                if label_coverage_status == "pass"
                else "OASIS-2 still has missing labels or label names, so supervised training must stay blocked."
            ),
            details={
                "unlabeled_row_count": unlabeled_row_count,
                "missing_label_name_count": missing_label_name_count,
            },
        )
    )
    if label_coverage_status == "fail":
        recommendations.append(
            "Fill `diagnosis_label` and `diagnosis_label_name` for every OASIS-2 session before training."
        )

    binary_label_status = "pass"
    binary_label_message = "OASIS-2 label values are compatible with the current binary classifier path."
    binary_label_details: dict[str, Any] = {}
    try:
        prepared_manifest = _prepare_supervised_manifest_frame(manifest_frame)
        group_frame = _build_group_frame(prepared_manifest, split_plan_frame)
        class_counts = prepared_manifest["label"].value_counts().sort_index().to_dict()
        dataset_summary["split_group_count"] = int(group_frame["split_group_key"].nunique())
        dataset_summary["mixed_label_group_count"] = int(group_frame["mixed_label_group"].sum())
        dataset_summary["session_label_distribution"] = {str(label): int(count) for label, count in class_counts.items()}
        dataset_summary["group_binary_label_distribution"] = {
            str(label): int(count)
            for label, count in group_frame["group_binary_label"].value_counts().sort_index().to_dict().items()
        }
        binary_label_details = {
            "session_label_distribution": dataset_summary["session_label_distribution"],
            "group_binary_label_distribution": dataset_summary["group_binary_label_distribution"],
            "mixed_label_group_count": dataset_summary["mixed_label_group_count"],
        }
    except OASIS2SupervisedError as error:
        prepared_manifest = None
        group_frame = None
        binary_label_status = "fail"
        binary_label_message = str(error)
        binary_label_details = {}

    checks.append(
        OASIS2TrainingReadinessCheck(
            name="binary_label_policy",
            status=binary_label_status,
            message=binary_label_message,
            details=binary_label_details,
        )
    )
    if binary_label_status == "fail":
        recommendations.append(
            "Keep OASIS-2 on the current binary label policy until a different model head and evaluation path are implemented."
        )

    if dataset_summary.get("mixed_label_group_count", 0) > 0:
        notes.append(
            "Some patient groups contain both binary labels across visits; this is allowed, and split stratification uses an ever-positive group label."
        )

    split_status = "fail"
    split_message = "OASIS-2 supervised split materialization was not attempted because earlier checks failed."
    split_details: dict[str, Any] = {}
    if label_coverage_status == "pass" and binary_label_status == "pass" and prepared_manifest is not None:
        try:
            artifacts = build_oasis2_supervised_split_artifacts(
                OASIS2SupervisedSplitConfig(
                    settings=resolved_settings,
                    manifest_path=resolved_manifest_path,
                    split_plan_path=resolved_split_plan_path,
                    seed=seed,
                    split_seed=split_seed,
                    train_fraction=train_fraction,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                )
            )
            split_status = "pass"
            split_message = "Subject-safe supervised OASIS-2 train/val/test manifests were materialized successfully."
            split_details = {
                "report_root": str(artifacts.report_root),
                "train_manifest_path": str(artifacts.train_manifest_path),
                "val_manifest_path": str(artifacts.val_manifest_path),
                "test_manifest_path": str(artifacts.test_manifest_path),
                "row_counts": dict(artifacts.summary_payload["row_counts"]),
                "subject_counts": dict(artifacts.summary_payload["subject_counts"]),
                "mixed_label_group_count": artifacts.summary_payload["mixed_label_group_count"],
            }
        except OASIS2SupervisedError as error:
            split_status = "fail"
            split_message = str(error)
            recommendations.append(
                "Adjust the split fractions or increase label coverage so each class has enough patient groups for stratified train/val/test splits."
            )

    checks.append(
        OASIS2TrainingReadinessCheck(
            name="supervised_split_materialization",
            status=split_status,
            message=split_message,
            details=split_details,
        )
    )

    if split_status == "pass":
        recommendations.append("Use the saved train/val/test manifests directly for OASIS-2 training runs.")
    else:
        recommendations.append("Do not start OASIS-2 training until the supervised split-materialization check passes.")

    return OASIS2TrainingReadinessReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        labeled_manifest_path=str(resolved_manifest_path),
        split_plan_path=str(resolved_split_plan_path),
        overall_status=_overall_status(checks),
        checks=checks,
        dataset_summary=dataset_summary,
        recommendations=recommendations,
        notes=notes,
    )


def save_oasis2_training_readiness_report(
    report: OASIS2TrainingReadinessReport,
    settings: AppSettings | None = None,
    *,
    file_stem: str = "oasis2_training_readiness",
) -> tuple[Path, Path]:
    """Save the OASIS-2 training-readiness report as JSON and Markdown."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "onboarding")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    payload = report.to_payload()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# OASIS-2 Training Readiness Report",
        "",
        f"- overall_status: {report.overall_status}",
        f"- generated_at: {report.generated_at}",
        f"- labeled_manifest_path: {report.labeled_manifest_path}",
        f"- split_plan_path: {report.split_plan_path}",
        "",
        "## Summary",
        "",
    ]
    summary = payload["summary"]
    lines.extend(
        [
            f"- pass: {summary.get('pass', 0)}",
            f"- warn: {summary.get('warn', 0)}",
            f"- fail: {summary.get('fail', 0)}",
            f"- labeled_row_count: {report.dataset_summary.get('labeled_row_count', 0)}",
            f"- unlabeled_row_count: {report.dataset_summary.get('unlabeled_row_count', 0)}",
            f"- unique_subject_count: {report.dataset_summary.get('unique_subject_count', 0)}",
            f"- split_group_count: {report.dataset_summary.get('split_group_count', 0)}",
            f"- mixed_label_group_count: {report.dataset_summary.get('mixed_label_group_count', 0)}",
            "",
            "## Checks",
            "",
        ]
    )
    lines.extend(f"- {check.status.upper()}: {check.name} - {check.message}" for check in report.checks)
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    if report.recommendations:
        lines.extend(["", "## Recommendations", ""])
        lines.extend(f"- {recommendation}" for recommendation in report.recommendations)

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
