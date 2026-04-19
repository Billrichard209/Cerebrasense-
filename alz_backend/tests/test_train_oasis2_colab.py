"""Tests for the OASIS-2 Colab bundle runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings
from src.data.oasis2_upload_bundle import build_oasis2_upload_bundle


def _load_oasis2_colab_module():
    """Load the OASIS-2 Colab runner script from disk."""

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_oasis2_colab.py"
    spec = importlib.util.spec_from_file_location("train_oasis2_colab", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for OASIS-2 bundle tests."""

    project_root = tmp_path / "seed_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    config_root = project_root / "configs"
    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path / "workspace",
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path / "kaggle",
        oasis_source_root=tmp_path / "OASIS",
        serving_config_path=config_root / "backend_serving.yaml",
    )


def _seed_oasis2_bundle_source(settings: AppSettings) -> Path:
    """Create a minimal valid OASIS-2 upload bundle from a fake source tree."""

    source_root = settings.collection_root
    hdr_path = source_root / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.hdr"
    img_path = hdr_path.with_suffix(".img")
    hdr_path.parent.mkdir(parents=True, exist_ok=True)
    hdr_path.write_text("hdr", encoding="utf-8")
    img_path.write_text("img", encoding="utf-8")

    interim_root = settings.data_root / "interim"
    interim_root.mkdir(parents=True, exist_ok=True)
    manifest_frame = pd.DataFrame(
        [
            {
                "image": str(hdr_path),
                "label": None,
                "label_name": None,
                "subject_id": "OAS2_0001",
                "session_id": "OAS2_0001_MR1",
                "visit_number": 1,
                "scan_timestamp": None,
                "dataset": "oasis2_raw",
                "dataset_type": "3d_volumes",
                "meta": json.dumps(
                    {
                        "paired_image": str(img_path),
                        "selected_acquisition_id": "mpr-1",
                        "acquisition_count": 1,
                    }
                ),
            }
        ]
    )
    manifest_frame.to_csv(interim_root / "oasis2_session_manifest.csv", index=False)
    manifest_frame.rename(columns={"image": "source_path", "visit_number": "visit_order"}).assign(
        record_type="oasis2_session"
    ).to_csv(interim_root / "oasis2_longitudinal_records.csv", index=False)
    pd.DataFrame(
        [{"subject_id": "OAS2_0001", "session_count": 1, "first_visit": 1, "last_visit": 1, "session_ids": "OAS2_0001_MR1"}]
    ).to_csv(interim_root / "oasis2_subject_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "image": str(hdr_path),
                "paired_image": str(img_path),
                "label": None,
                "label_name": None,
                "subject_id": "OAS2_0001",
                "session_id": "OAS2_0001_MR1",
                "visit_number": 1,
                "scan_timestamp": None,
                "dataset": "oasis2_raw",
                "dataset_type": "3d_volumes",
                "source_part": "OAS2_RAW_PART1",
                "acquisition_id": "mpr-1",
                "volume_format": "analyze_pair",
                "meta": "{}",
            }
        ]
    ).to_csv(interim_root / "oasis2_raw_inventory.csv", index=False)
    (interim_root / "oasis2_raw_inventory_dropped_rows.csv").write_text("reason\n", encoding="utf-8")
    (interim_root / "oasis2_raw_inventory_summary.json").write_text("{}", encoding="utf-8")
    (interim_root / "oasis2_session_manifest_summary.json").write_text("{}", encoding="utf-8")
    (
        interim_root / "oasis2_metadata_template.csv"
    ).write_text(
        "\n".join(
            [
                "subject_id,session_id,visit_number,scan_timestamp,age_at_visit,sex,clinical_status,cdr_global,mmse,diagnosis_label,diagnosis_label_name,metadata_source,metadata_complete,split_group_hint,notes",
                "OAS2_0001,OAS2_0001_MR1,1,,,,,,,,,,False,OAS2_0001,",
            ]
        ),
        encoding="utf-8",
    )
    (interim_root / "oasis2_metadata_template_summary.json").write_text("{}", encoding="utf-8")
    (interim_root / "oasis2_labeled_prep_manifest.csv").write_text("image,label\n", encoding="utf-8")
    (
        interim_root / "oasis2_subject_safe_split_plan.csv"
    ).write_text(
        "split_group_hint,subject_ids,primary_subject_id,session_count,visit_count,metadata_row_count,candidate_label_row_count,subject_safe_bucket,future_role_hint\n"
        "OAS2_0001,OAS2_0001,OAS2_0001,1,1,1,0,0,holdout_candidate\n",
        encoding="utf-8",
    )
    (interim_root / "oasis2_subject_safe_split_plan_summary.json").write_text("{}", encoding="utf-8")

    readiness_root = settings.outputs_root / "reports" / "readiness"
    readiness_root.mkdir(parents=True, exist_ok=True)
    (readiness_root / "oasis2_readiness.json").write_text("{}", encoding="utf-8")
    (readiness_root / "oasis2_readiness.md").write_text("# ready", encoding="utf-8")
    onboarding_root = settings.outputs_root / "reports" / "onboarding"
    onboarding_root.mkdir(parents=True, exist_ok=True)
    (onboarding_root / "oasis2_adapter_status.json").write_text("{}", encoding="utf-8")
    (onboarding_root / "oasis2_adapter_status.md").write_text("# adapter", encoding="utf-8")
    (onboarding_root / "oasis2_metadata_adapter_status.json").write_text("{}", encoding="utf-8")
    (onboarding_root / "oasis2_metadata_adapter_status.md").write_text("# metadata", encoding="utf-8")
    (onboarding_root / "oasis2_subject_safe_split_plan.md").write_text("# split", encoding="utf-8")

    result = build_oasis2_upload_bundle(
        settings=settings,
        source_root=source_root,
        materialize_mode="copy",
        output_root=source_root / "bundle",
    )
    return result.bundle_root


def _write_official_demographics(path: Path) -> Path:
    frame = pd.DataFrame(
        [
            {
                "Subject ID": "OAS2_0001",
                "MRI ID": "OAS2_0001_MR1",
                "Group": "Converted",
                "Visit": 1,
                "MR Delay": 0,
                "M/F": "M",
                "Hand": "R",
                "Age": 87,
                "EDUC": 14,
                "SES": 2.0,
                "MMSE": 27.0,
                "CDR": 0.0,
                "eTIV": 1986.55,
                "nWBV": 0.696106,
                "ASF": 0.88344,
            }
        ]
    )
    frame.to_excel(path, index=False)
    return path


def test_oasis2_colab_pipeline_blocks_cleanly_when_labels_are_missing(tmp_path: Path) -> None:
    """A valid uploaded bundle should stop cleanly at the label gate instead of attempting training."""

    module = _load_oasis2_colab_module()
    bundle_root = _seed_oasis2_bundle_source(_build_settings(tmp_path))
    runtime_root = tmp_path / "runtime"

    args = module.build_parser().parse_args(
        [
            "--bundle-root",
            str(bundle_root),
            "--runtime-root",
            str(runtime_root),
            "--no-auto-fill-from-official-demographics",
        ]
    )
    summary = module.run_oasis2_colab_pipeline(args)

    assert summary["training_ready"] is False
    assert summary["training_started"] is False
    assert summary["readiness_status"] == "fail"
    assert "label_coverage" in summary["blocked_reason"]
    assert Path(summary["runtime_metadata_template_path"]).exists()
    assert Path(summary["training_readiness_json_path"]).exists()
    assert Path(summary["summary_json_path"]).exists()


def test_oasis2_colab_pipeline_can_require_training_ready(tmp_path: Path) -> None:
    """The strict flag should surface the blocked training state as a nonzero exit."""

    module = _load_oasis2_colab_module()
    bundle_root = _seed_oasis2_bundle_source(_build_settings(tmp_path))
    runtime_root = tmp_path / "runtime"

    args = module.build_parser().parse_args(
        [
            "--bundle-root",
            str(bundle_root),
            "--runtime-root",
            str(runtime_root),
            "--require-training-ready",
            "--no-auto-fill-from-official-demographics",
        ]
    )

    try:
        module.run_oasis2_colab_pipeline(args)
    except SystemExit as exc:
        assert "label_coverage" in str(exc)
    else:
        raise AssertionError("Expected strict OASIS-2 Colab readiness gate to stop the pipeline.")


def test_oasis2_colab_pipeline_can_autofill_runtime_metadata_from_official_demographics(tmp_path: Path) -> None:
    """When given the official demographics sheet, the Colab runner should fill the runtime template automatically."""

    module = _load_oasis2_colab_module()
    bundle_root = _seed_oasis2_bundle_source(_build_settings(tmp_path))
    runtime_root = tmp_path / "runtime"
    demographics_path = _write_official_demographics(tmp_path / "oasis2_demographics.xlsx")

    args = module.build_parser().parse_args(
        [
            "--bundle-root",
            str(bundle_root),
            "--runtime-root",
            str(runtime_root),
            "--demographics-path",
            str(demographics_path),
        ]
    )
    summary = module.run_oasis2_colab_pipeline(args)

    runtime_template = pd.read_csv(summary["runtime_metadata_template_path"])
    assert runtime_template.loc[0, "diagnosis_label"] == 0
    assert runtime_template.loc[0, "diagnosis_label_name"] == "nondemented"
    assert Path(summary["official_demographics_import_json_path"]).exists()
    assert summary["official_demographics_import_summary"]["matched_row_count"] == 1
    assert "label_coverage" not in str(summary["blocked_reason"])
