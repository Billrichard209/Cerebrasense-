"""Check reproducible OASIS splits and optional MONAI dataloader construction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import OASISLoaderConfig, build_oasis_dataloaders, build_oasis_datasets


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build reproducible OASIS subject-safe splits and optionally instantiate MONAI dataloaders."
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--weighted-sampling", action="store_true")
    parser.add_argument(
        "--build-dataloaders",
        action="store_true",
        help="Also build MONAI dataloaders after generating the split manifests.",
    )
    return parser


def _make_config(args: argparse.Namespace) -> OASISLoaderConfig:
    """Convert CLI arguments to the loader config."""

    return OASISLoaderConfig(
        manifest_path=args.manifest_path,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        weighted_sampling=args.weighted_sampling,
    )


def _print_summary(summary_path: Path) -> None:
    """Print a compact summary from the saved split report."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"report_root={summary_path.parent}")
    print(f"summary={summary_path}")
    print(f"subject_counts={summary['subject_counts']}")
    print(f"row_counts={summary['row_counts']}")
    print(f"label_distribution_by_split={summary['label_distribution_by_split']}")
    print(f"subject_overlap={summary['subject_overlap']}")
    print(f"longitudinal={summary['longitudinal']}")


def main() -> None:
    """Build and validate the OASIS split artifacts."""

    args = build_parser().parse_args()
    cfg = _make_config(args)

    if args.build_dataloaders or args.weighted_sampling:
        bundle = build_oasis_dataloaders(cfg)
        artifacts = bundle.dataset_bundle.split_artifacts
        print("dataloaders_built=true")
        print(f"weighted_sampler={bundle.train_sampler is not None}")
    else:
        bundle = build_oasis_datasets(cfg)
        artifacts = bundle.split_artifacts
        print("dataloaders_built=false")
        print("weighted_sampler=false")

    print(f"split_assignments={artifacts.split_assignments_path}")
    print(f"longitudinal_index={artifacts.longitudinal_index_path}")
    print(f"longitudinal_subject_summary={artifacts.longitudinal_subject_summary_path}")
    print(f"train_manifest={artifacts.train_manifest_path}")
    print(f"val_manifest={artifacts.val_manifest_path}")
    print(f"test_manifest={artifacts.test_manifest_path}")
    _print_summary(artifacts.summary_path)


if __name__ == "__main__":
    main()
