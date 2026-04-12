"""Run checkpoint-backed OASIS inference for one MRI volume."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict_oasis import OASISInferenceConfig, predict_oasis_checkpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Run OASIS-1 checkpoint inference on one MRI volume.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--output-name", type=str, default="oasis_prediction")
    parser.add_argument("--model-config-path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--scan-timestamp", type=str, default=None)
    parser.add_argument("--no-report", action="store_true")
    return parser


def main() -> None:
    """Run inference and print the important output fields."""

    args = build_parser().parse_args()
    result = predict_oasis_checkpoint(
        OASISInferenceConfig(
            checkpoint_path=args.checkpoint_path,
            image_path=args.image_path,
            output_name=args.output_name,
            model_config_path=args.model_config_path,
            threshold=args.threshold,
            image_size=tuple(args.image_size),
            device=args.device,
            subject_id=args.subject_id,
            session_id=args.session_id,
            scan_timestamp=args.scan_timestamp,
            save_report=not args.no_report,
        )
    )
    print(f"predicted_label={result.prediction.predicted_index}")
    print(f"predicted_label_name={result.prediction.label}")
    print(f"confidence={result.prediction.confidence}")
    print(f"probabilities={result.prediction.probabilities}")
    print(f"prediction_json={result.prediction_json_path}")
    print(f"summary_report={result.summary_report_path}")


if __name__ == "__main__":
    main()
