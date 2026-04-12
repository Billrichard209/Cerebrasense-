"""Run the clean backend scan inference pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.calibration import ConfidenceBandConfig  # noqa: E402
from src.inference.pipeline import PredictScanOptions, predict_scan  # noqa: E402
from src.models.registry import load_current_oasis_model_entry  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Predict Alzheimer decision-support score for one MRI scan.")
    parser.add_argument("--scan-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None, help="MONAI preprocessing YAML path.")
    parser.add_argument("--model-config-path", type=Path, default=None)
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--use-registry-threshold", action="store_true")
    parser.add_argument("--output-name", type=str, default="scan_prediction")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-debug-slices", action="store_true")
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--scan-timestamp", type=str, default=None)
    return parser


def main() -> None:
    """Run prediction and print compact result metadata."""

    args = build_parser().parse_args()
    registry_entry = load_current_oasis_model_entry(args.registry_path) if args.registry_path or args.checkpoint_path is None else None
    checkpoint_path = args.checkpoint_path or (registry_entry.checkpoint_path if registry_entry is not None else None)
    if checkpoint_path is None:
        raise ValueError("Provide --checkpoint-path or a loadable model registry entry.")
    config_path = args.config_path or (registry_entry.preprocessing_config_path if registry_entry is not None else None)
    model_config_path = args.model_config_path or (
        Path(registry_entry.model_config_path) if registry_entry is not None and registry_entry.model_config_path else None
    )
    threshold = (
        float(registry_entry.recommended_threshold)
        if registry_entry is not None and args.use_registry_threshold
        else args.threshold
    )
    confidence_config = None
    if registry_entry is not None:
        policy = dict(registry_entry.confidence_policy)
        scaling = dict(registry_entry.temperature_scaling)
        confidence_config = ConfidenceBandConfig(
            temperature=float(scaling.get("temperature", 1.0)),
            high_confidence_min=float(policy.get("high_confidence_min", 0.85)),
            medium_confidence_min=float(policy.get("medium_confidence_min", 0.65)),
            high_entropy_max=float(policy.get("high_entropy_max", 0.35)),
            medium_entropy_max=float(policy.get("medium_entropy_max", 0.70)),
        )
    payload = predict_scan(
        args.scan_path,
        checkpoint_path,
        config_path,
        options=PredictScanOptions(
            output_name=args.output_name,
            threshold=threshold,
            device=args.device,
            model_config_path=model_config_path,
            save_debug_slices=args.save_debug_slices,
            subject_id=args.subject_id,
            session_id=args.session_id,
            scan_timestamp=args.scan_timestamp,
            confidence_config=confidence_config,
        ),
    )
    print(f"predicted_label={payload['predicted_label']}")
    print(f"label_name={payload['label_name']}")
    print(f"probability_score={payload['probability_score']}")
    print(f"confidence_score={payload['confidence_score']}")
    print(f"confidence_level={payload['confidence_level']}")
    print(f"review_flag={payload['review_flag']}")
    print(f"prediction_json={payload['outputs']['prediction_json']}")
    print("payload=" + json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
