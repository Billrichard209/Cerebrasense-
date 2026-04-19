"""Tests for supervised OASIS-2 split materialization and readiness checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings
from src.data.oasis2_supervised import (
    OASIS2SupervisedSplitConfig,
    build_oasis2_supervised_split_artifacts,
    build_oasis2_training_readiness_report,
    save_oasis2_training_readiness_report,
)


def _settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path,
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path / "kaggle",
        oasis_source_root=tmp_path / "OASIS",
    )


def _write_supervised_inputs(settings: AppSettings, *, missing_label: bool = False) -> None:
    manifest_path = settings.data_root / "interim" / "oasis2_labeled_prep_manifest.csv"
    split_plan_path = settings.data_root / "interim" / "oasis2_subject_safe_split_plan.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    positive_subjects = {"OAS2_0005", "OAS2_0006", "OAS2_0007", "OAS2_0008"}
    mixed_subject = "OAS2_0008"
    for index in range(1, 9):
        subject_id = f"OAS2_{index:04d}"
        for visit_number in (1, 2):
            image_path = (
                settings.collection_root
                / "dataset"
                / "OAS2_RAW_PART1"
                / f"{subject_id}_MR{visit_number}"
                / "RAW"
                / "mpr-1.nifti.hdr"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"hdr")
            if subject_id in positive_subjects:
                label = 1
                label_name = "demented_like"
            else:
                label = 0
                label_name = "nondemented_like"
            if subject_id == mixed_subject and visit_number == 1:
                label = 0
                label_name = "nondemented_like"
            if missing_label and subject_id == "OAS2_0002" and visit_number == 2:
                label = None
                label_name = None
            manifest_rows.append(
                {
                    "image": str(image_path),
                    "label": label,
                    "label_name": label_name,
                    "subject_id": subject_id,
                    "session_id": f"{subject_id}_MR{visit_number}",
                    "visit_number": visit_number,
                    "scan_timestamp": f"2024-0{visit_number}-01",
                    "dataset": "oasis2_raw",
                    "dataset_type": "3d_volumes",
                    "meta": json.dumps(
                        {
                            "oasis2_metadata": {
                                "split_group_hint": subject_id,
                                "metadata_complete": True,
                            }
                        }
                    ),
                }
            )
        split_rows.append(
            {
                "split_group_hint": subject_id,
                "subject_ids": subject_id,
                "primary_subject_id": subject_id,
                "session_count": 2,
                "visit_count": 2,
                "metadata_row_count": 2,
                "candidate_label_row_count": 2,
                "subject_safe_bucket": index % 5,
                "future_role_hint": "holdout_candidate" if index % 5 == 0 else "development_candidate",
            }
        )

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(split_rows).to_csv(split_plan_path, index=False)


def test_build_oasis2_supervised_split_artifacts_materializes_subject_safe_manifests(tmp_path: Path) -> None:
    """The supervised split builder should save disjoint patient-safe train/val/test manifests."""

    settings = _settings(tmp_path)
    _write_supervised_inputs(settings)

    artifacts = build_oasis2_supervised_split_artifacts(
        OASIS2SupervisedSplitConfig(
            settings=settings,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            split_seed=7,
        )
    )

    assert artifacts.train_manifest_path.exists()
    assert artifacts.val_manifest_path.exists()
    assert artifacts.test_manifest_path.exists()
    assert artifacts.summary_json_path.exists()
    assert artifacts.summary_md_path.exists()
    assert artifacts.summary_payload["row_counts"] == {"train": 8, "val": 4, "test": 4}
    assert artifacts.summary_payload["group_overlap"] == {"train_val": [], "train_test": [], "val_test": []}
    assert artifacts.summary_payload["subject_overlap"] == {"train_val": [], "train_test": [], "val_test": []}
    assert artifacts.summary_payload["mixed_label_group_count"] == 1


def test_oasis2_training_readiness_fails_without_complete_labels(tmp_path: Path) -> None:
    """Training readiness should fail loudly when labels are still incomplete."""

    settings = _settings(tmp_path)
    _write_supervised_inputs(settings, missing_label=True)

    report = build_oasis2_training_readiness_report(
        settings=settings,
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        split_seed=7,
    )

    checks = {check.name: check for check in report.checks}
    assert report.overall_status == "fail"
    assert checks["label_coverage"].status == "fail"
    assert report.dataset_summary["unlabeled_row_count"] == 1


def test_oasis2_training_readiness_passes_and_can_be_saved(tmp_path: Path) -> None:
    """A fully labeled binary OASIS-2 prep set should pass readiness and write reports."""

    settings = _settings(tmp_path)
    _write_supervised_inputs(settings)

    report = build_oasis2_training_readiness_report(
        settings=settings,
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        split_seed=7,
    )
    json_path, md_path = save_oasis2_training_readiness_report(
        report,
        settings,
        file_stem="unit_oasis2_training_readiness",
    )

    checks = {check.name: check for check in report.checks}
    assert report.overall_status == "pass"
    assert checks["supervised_split_materialization"].status == "pass"
    assert json.loads(json_path.read_text(encoding="utf-8"))["overall_status"] == "pass"
    assert "OASIS-2 Training Readiness Report" in md_path.read_text(encoding="utf-8")
