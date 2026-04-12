"""Build an external-cohort manifest from 3D MRI files and optional metadata CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.external_cohort import summarize_external_cohort_manifest  # noqa: E402
from src.data.external_manifest_builder import (  # noqa: E402
    SUPPORTED_3D_IMAGE_GLOBS,
    ExternalManifestBuildError,
    ExternalManifestBuilderConfig,
    build_external_cohort_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build an external-cohort manifest for a true outside 3D MRI dataset. "
            "No automatic label harmonization is applied."
        )
    )
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dataset-type", type=str, default="3d_volumes")
    parser.add_argument("--metadata-csv-path", type=Path, default=None)
    parser.add_argument("--image-column", type=str, default=None)
    parser.add_argument("--label-column", type=str, default=None)
    parser.add_argument("--label-name-column", type=str, default=None)
    parser.add_argument("--subject-id-column", type=str, default=None)
    parser.add_argument("--session-id-column", type=str, default=None)
    parser.add_argument("--scan-timestamp-column", type=str, default=None)
    parser.add_argument("--meta-columns", nargs="*", default=())
    parser.add_argument("--image-globs", nargs="*", default=list(SUPPORTED_3D_IMAGE_GLOBS))
    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument("--require-labels", action="store_true")
    parser.add_argument("--validate-for-evaluation", action="store_true")
    return parser


def main() -> None:
    """Build the manifest, save a build report, and optionally validate it for evaluation."""

    args = build_parser().parse_args()
    try:
        result = build_external_cohort_manifest(
            ExternalManifestBuilderConfig(
                images_root=args.images_root,
                dataset_name=args.dataset_name,
                output_path=args.output_path,
                dataset_type=args.dataset_type,
                metadata_csv_path=args.metadata_csv_path,
                image_column=args.image_column,
                label_column=args.label_column,
                label_name_column=args.label_name_column,
                subject_id_column=args.subject_id_column,
                session_id_column=args.session_id_column,
                scan_timestamp_column=args.scan_timestamp_column,
                meta_columns=tuple(args.meta_columns),
                image_globs=tuple(args.image_globs),
                recursive=not args.non_recursive,
                require_labels=args.require_labels,
            )
        )

        print(f"manifest_path={result.manifest_path}")
        print(f"build_report_path={result.report_path}")
        print(f"row_count={result.row_count}")
        print(f"discovered_image_count={result.discovered_image_count}")
        print(f"matched_image_count={result.matched_image_count}")
        print(f"unmatched_image_count={result.unmatched_image_count}")
        if result.warnings:
            print(f"warnings={json.dumps(result.warnings)}")

        if args.validate_for_evaluation:
            summary = summarize_external_cohort_manifest(
                result.manifest_path,
                require_labels=True,
                expected_dataset_type="3d_volumes",
            )
            print(f"validation_manifest_hash={summary.manifest_hash_sha256}")
            print(f"validation_dataset_name={summary.dataset_name}")
            print(f"validation_sample_count={summary.sample_count}")
            print(f"validation_subject_count={summary.subject_count}")
    except (FileNotFoundError, ExternalManifestBuildError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print(
            "Hint: replace example placeholders like 'path\\\\to\\\\ADNI_scans' and "
            "'path\\\\to\\\\ADNI_metadata.csv' with real paths on your machine.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
