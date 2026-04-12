"""Generate Grad-CAM-style explanation artifacts for one MRI scan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.gradcam import ExplainScanConfig, explain_scan  # noqa: E402


def _parse_slice_indices(raw_value: str | None) -> tuple[int, ...] | None:
    """Parse comma-separated slice indices."""

    if raw_value is None or not raw_value.strip():
        return None
    return tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Generate OASIS Grad-CAM-style explanation overlays.")
    parser.add_argument("--scan-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--preprocessing-config-path", type=Path, default=None)
    parser.add_argument("--model-config-path", type=Path, default=None)
    parser.add_argument("--output-name", type=str, default="scan_explanation")
    parser.add_argument("--target-layer", type=str, default="auto")
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image-size", type=int, nargs=3, default=None)
    parser.add_argument("--slice-axis", choices=["axial", "coronal", "sagittal"], default="axial")
    parser.add_argument("--slice-indices", type=str, default=None, help="Comma-separated slice indices, e.g. 32,48,64.")
    parser.add_argument("--no-saliency", action="store_true")
    parser.add_argument("--true-label", type=int, default=None)
    return parser


def main() -> None:
    """Run explanation and print artifact paths."""

    args = build_parser().parse_args()
    result = explain_scan(
        ExplainScanConfig(
            scan_path=args.scan_path,
            checkpoint_path=args.checkpoint_path,
            preprocessing_config_path=args.preprocessing_config_path,
            model_config_path=args.model_config_path,
            output_name=args.output_name,
            target_layer=args.target_layer,
            target_class=args.target_class,
            device=args.device,
            image_size=tuple(args.image_size) if args.image_size is not None else None,
            slice_axis=args.slice_axis,
            slice_indices=_parse_slice_indices(args.slice_indices),
            save_saliency=not args.no_saliency,
            true_label=args.true_label,
        )
    )
    print(f"output_root={result.output_root}")
    print(f"report_json={result.report_path}")
    print(f"gradcam_overlays={[str(path) for path in result.overlay_paths]}")
    print(f"saliency_overlays={[str(path) for path in result.saliency_paths]}")
    print(f"target_layer={result.payload['target_layer']}")
    print(f"target_class={result.payload['target_class']}")
    print(f"confidence_level={result.payload['confidence_level']}")
    print(f"explanation_quality={result.payload['explanation_quality']}")


if __name__ == "__main__":
    main()
