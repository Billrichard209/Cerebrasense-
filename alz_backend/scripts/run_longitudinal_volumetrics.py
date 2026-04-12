"""Run subject-level longitudinal structural proxy analysis for OASIS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pandas import isna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import get_app_settings
from src.data.oasis_dataset import load_oasis_manifest
from src.longitudinal.structural import (
    build_oasis_structural_longitudinal_summary,
    save_structural_longitudinal_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Run subject-level OASIS structural longitudinal analysis.")
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--max-timepoints", type=int, default=None)
    parser.add_argument("--output-stem", type=str, default=None)
    return parser


def _resolve_subject_id(subject_id: str | None, *, split: str | None, manifest_path: Path | None) -> str:
    """Use the requested subject ID or fall back to the first available OASIS subject."""

    if subject_id:
        return subject_id
    settings = get_app_settings()
    frame = load_oasis_manifest(settings, split=split, manifest_path=manifest_path)
    for value in frame["subject_id"]:
        if not isna(value) and str(value).strip():
            return str(value).strip()
    raise ValueError("Could not resolve an OASIS subject_id from the manifest.")


def main() -> None:
    """Run the subject-level analysis and save the report."""

    args = build_parser().parse_args()
    subject_id = _resolve_subject_id(args.subject_id, split=args.split, manifest_path=args.manifest_path)
    summary = build_oasis_structural_longitudinal_summary(
        subject_id,
        split=args.split,
        manifest_path=args.manifest_path,
        max_timepoints=args.max_timepoints,
    )
    output_path = save_structural_longitudinal_report(
        summary,
        file_stem=args.output_stem or f"{subject_id}_structural_longitudinal",
    )

    print(f"report={output_path}")
    print(f"subject_id={summary.subject_id}")
    print(f"timepoint_count={summary.timepoint_count}")
    print(f"change_count={len(summary.changes)}")
    print(f"warnings={summary.warnings}")


if __name__ == "__main__":
    main()
