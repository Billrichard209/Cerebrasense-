"""Run a limited OASIS baseline model evaluation and save probability metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import OASISLoaderConfig, build_oasis_dataloaders
from src.evaluation.evaluate_oasis import evaluate_oasis_model_on_loader, save_oasis_evaluation_report
from src.inference.serving import resolve_oasis_decision_policy
from src.evaluation.oasis_run import load_oasis_checkpoint
from src.models.factory import build_model, load_oasis_model_config
from src.transforms.oasis_transforms import OASISSpatialConfig, OASISTransformConfig, load_oasis_transform_config


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Evaluate the OASIS MONAI baseline on a small validation subset.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", type=str, default="oasis_baseline_evaluation")
    parser.add_argument("--use-active-serving-policy", action="store_true")
    parser.add_argument("--serving-config-path", type=Path, default=None)
    parser.add_argument("--registry-path", type=Path, default=None)
    return parser


def _load_checkpoint_if_requested(model: object, checkpoint_path: Path | None, *, device: str) -> object:
    """Load a PyTorch checkpoint into the model when supplied."""

    if checkpoint_path is None:
        return model
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    loaded_checkpoint = load_oasis_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(loaded_checkpoint.model_state_dict)
    return model


def main() -> None:
    """Run the evaluation and print the saved metrics path."""

    args = build_parser().parse_args()
    model_cfg = load_oasis_model_config(args.model_config)
    model = build_model(model_cfg)
    model = _load_checkpoint_if_requested(model, args.checkpoint_path, device=args.device)

    transform_cfg = load_oasis_transform_config()
    transform_cfg = OASISTransformConfig(
        load=transform_cfg.load,
        orientation=transform_cfg.orientation,
        spacing=transform_cfg.spacing,
        intensity=transform_cfg.intensity,
        skull_strip=transform_cfg.skull_strip,
        spatial=OASISSpatialConfig(spatial_size=tuple(args.image_size)),
        augmentation=transform_cfg.augmentation,
    )
    loader_cfg = OASISLoaderConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        transform_config=transform_cfg,
    )
    dataloaders = build_oasis_dataloaders(loader_cfg)
    loader = dataloaders.val_loader if args.split == "val" else dataloaders.test_loader
    if args.use_active_serving_policy or args.registry_path is not None or args.serving_config_path is not None:
        decision_policy = resolve_oasis_decision_policy(
            explicit_threshold=args.threshold,
            serving_config_path=args.serving_config_path,
            registry_path=args.registry_path,
        )
        calibration_config = decision_policy.confidence_config
        decision_threshold = decision_policy.threshold
    else:
        calibration_config = None
        decision_threshold = float(args.threshold if args.threshold is not None else 0.5)
    result = evaluate_oasis_model_on_loader(
        model=model,
        loader=loader,
        device=args.device,
        class_names=model_cfg.class_names,
        max_batches=args.max_batches,
        calibration_config=calibration_config,
        decision_threshold=decision_threshold,
    )
    output_path = save_oasis_evaluation_report(result, run_name=args.output_name)

    print(f"metrics={output_path}")
    print(f"split={args.split}")
    print(f"sample_count={result.metrics['sample_count']}")
    print(f"accuracy={result.metrics['accuracy']}")
    print(f"mean_confidence={result.metrics['mean_confidence']}")
    print(f"mean_normalized_entropy={result.metrics['mean_normalized_entropy']}")
    if args.checkpoint_path is None:
        print("warning=No checkpoint was supplied, so this evaluates random baseline weights.")


if __name__ == "__main__":
    main()
