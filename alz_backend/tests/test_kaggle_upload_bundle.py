"""Tests for the Kaggle upload bundle builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings
from src.data.kaggle_upload_bundle import build_kaggle_upload_bundle


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for upload bundle tests."""

    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    config_root = project_root / "configs"
    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path,
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path,
        oasis_source_root=tmp_path / "OASIS",
        serving_config_path=config_root / "backend_serving.yaml",
    )


def test_build_kaggle_upload_bundle_materializes_expected_subset(tmp_path: Path) -> None:
    """The bundle builder should create a portable Kaggle bundle with references."""

    settings = _build_settings(tmp_path)
    original_image = settings.kaggle_source_root / "OriginalDataset" / "NonDemented" / "sample_1.jpg"
    augmented_image = settings.kaggle_source_root / "AugmentedAlzheimerDataset" / "MildDemented" / "sample_2.jpg"
    original_image.parent.mkdir(parents=True, exist_ok=True)
    augmented_image.parent.mkdir(parents=True, exist_ok=True)
    original_image.write_text("img1", encoding="utf-8")
    augmented_image.write_text("img2", encoding="utf-8")

    interim_root = settings.data_root / "interim"
    interim_root.mkdir(parents=True, exist_ok=True)
    manifest_frame = pd.DataFrame(
        [
            {
                "image": str(original_image),
                "label": 0,
                "label_name": "NonDemented",
                "subject_id": "",
                "scan_timestamp": "",
                "dataset": "kaggle_alz",
                "dataset_type": "2d_slices",
                "meta": json.dumps({"subset": "OriginalDataset", "source_root": str(settings.kaggle_source_root)}),
            },
            {
                "image": str(augmented_image),
                "label": 1,
                "label_name": "MildDemented",
                "subject_id": "",
                "scan_timestamp": "",
                "dataset": "kaggle_alz",
                "dataset_type": "2d_slices",
                "meta": json.dumps(
                    {"subset": "AugmentedAlzheimerDataset", "source_root": str(settings.kaggle_source_root)}
                ),
            },
        ]
    )
    manifest_frame.to_csv(interim_root / "kaggle_alz_manifest.csv", index=False)
    manifest_frame.iloc[[0]].to_csv(interim_root / "kaggle_alz_train_manifest.csv", index=False)
    manifest_frame.iloc[[1]].to_csv(interim_root / "kaggle_alz_val_manifest.csv", index=False)
    manifest_frame.iloc[[0]].to_csv(interim_root / "kaggle_alz_test_manifest.csv", index=False)
    (interim_root / "kaggle_alz_split_assignments.csv").write_text(
        "image,label,label_name,subject_id,dataset_type,subset_source,group_id,group_kind,split,assignment_reason\n"
        f"{original_image},0,NonDemented,,2d_slices,OriginalDataset,row::0,row,train,base_split\n",
        encoding="utf-8",
    )
    (interim_root / "kaggle_alz_manifest_summary.json").write_text("{}", encoding="utf-8")
    (interim_root / "kaggle_alz_manifest_dropped_rows.csv").write_text("image,reason\n", encoding="utf-8")
    (interim_root / "kaggle_alz_split_summary.json").write_text("{}", encoding="utf-8")

    result = build_kaggle_upload_bundle(
        settings=settings,
        materialize_mode="copy",
        output_root=tmp_path / "bundle",
    )

    copied_original = result.bundle_root / "OriginalDataset" / "NonDemented" / "sample_1.jpg"
    copied_augmented = result.bundle_root / "AugmentedAlzheimerDataset" / "MildDemented" / "sample_2.jpg"
    assert copied_original.exists()
    assert copied_augmented.exists()
    assert result.materialized_file_count == 2
    assert result.missing_reference_count == 0
    assert set(result.included_subset_names) == {"AugmentedAlzheimerDataset", "OriginalDataset"}

    relative_manifest = pd.read_csv(result.relative_manifest_path)
    assert relative_manifest.loc[0, "image"].startswith("OriginalDataset/")
    assert (result.bundle_root / "backend_reference" / "kaggle_alz_train_manifest_relative.csv").exists()
    assert (result.bundle_root / "backend_reference" / "kaggle_alz_split_assignments_relative.csv").exists()
    assert (result.bundle_root / "README.md").exists()
