"""Run foreground-proxy volumetric analysis for one OASIS MRI volume."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.volumetrics.service import analyze_oasis_volume, save_oasis_volumetric_report


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Run foreground-proxy volumetric analysis for one OASIS MRI volume.")
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--scan-timestamp", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--output-stem", type=str, default=None)
    return parser


def main() -> None:
    """Run the analysis and save the report."""

    args = build_parser().parse_args()
    result = analyze_oasis_volume(
        image_path=args.image_path,
        subject_id=args.subject_id,
        session_id=args.session_id,
        scan_timestamp=args.scan_timestamp,
        split=args.split,
        row_index=args.row_index,
        manifest_path=args.manifest_path,
    )
    report = result.to_report_payload()
    output_path = save_oasis_volumetric_report(
        report,
        file_stem=args.output_stem or (result.session_id or result.image_path.stem),
    )

    print(f"report={output_path}")
    print(f"image={result.image_path}")
    print(f"subject_id={result.subject_id}")
    print(f"session_id={result.session_id}")
    print(f"foreground_voxels={result.foreground_voxels}")
    print(f"foreground_proxy_brain_mm3={report['measurements'][0]['value_mm3']}")
    print(f"warnings={result.warnings}")


if __name__ == "__main__":
    main()
