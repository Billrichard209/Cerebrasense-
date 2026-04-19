"""Tests for OASIS-2 dataloader record normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.oasis2_loaders import _records_from_split_frame


def test_records_from_split_frame_normalizes_optional_text_for_collation(tmp_path: Path) -> None:
    """Optional string fields should not remain `None` by the time MONAI collates a batch."""

    image_path = tmp_path / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.hdr"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"hdr")

    frame = pd.DataFrame(
        [
            {
                "image": str(image_path),
                "label": 0,
                "label_name": None,
                "subject_id": "OAS2_0001",
                "session_id": "OAS2_0001_MR1",
                "visit_number": 1,
                "scan_timestamp": None,
                "dataset": None,
                "dataset_type": None,
                "split_group_key": "OAS2_0001",
                "group_binary_label": 0,
                "mixed_label_group": False,
                "subject_safe_bucket": 0,
                "future_role_hint": None,
                "meta": json.dumps({"source_part": "OAS2_RAW_PART1"}),
            }
        ]
    )

    records = _records_from_split_frame(frame)

    assert records[0]["label_name"] == ""
    assert records[0]["scan_timestamp"] == ""
    assert records[0]["dataset"] == ""
    assert records[0]["dataset_type"] == ""
    assert records[0]["meta"]["future_role_hint"] == ""
