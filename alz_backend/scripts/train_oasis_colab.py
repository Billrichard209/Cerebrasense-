"""Colab-friendly OASIS training pipeline with persistent runtime roots."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd


def _running_in_colab() -> bool:
    """Return whether the current interpreter looks like Google Colab."""

    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _mount_drive_if_requested(should_mount: bool) -> None:
    """Mount Google Drive when running inside Colab and requested."""

    if not should_mount or not _running_in_colab():
        return
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def _resolve_existing_oasis_root(source_root: Path) -> Path:
    """Return the concrete OASIS source folder, tolerating a nested `OASIS/`."""

    resolved = source_root.expanduser().resolve()
    nested = resolved / "OASIS"
    return nested if nested.exists() else resolved


def _resolve_training_device(requested_device: str) -> tuple[str, bool]:
    """Resolve a safe device and mixed-precision setting for the current runtime."""

    import torch

    normalized = requested_device.strip().lower()
    if normalized == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = normalized

    if resolved == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA, but this runtime does not expose a GPU. Falling back to CPU.")
        resolved = "cpu"

    return resolved, resolved == "cuda"


def _configure_runtime_roots(*, project_root: Path, runtime_root: Path | None) -> tuple[Path, Path]:
    """Resolve and export the runtime `data/` and `outputs/` roots."""

    if runtime_root is None:
        data_root = project_root / "data"
        outputs_root = project_root / "outputs"
    else:
        resolved_runtime_root = runtime_root.expanduser().resolve()
        data_root = resolved_runtime_root / "data"
        outputs_root = resolved_runtime_root / "outputs"

    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    os.environ["ALZ_DATA_ROOT"] = str(data_root)
    os.environ["ALZ_OUTPUTS_ROOT"] = str(outputs_root)
    return data_root, outputs_root


def _stage_oasis_source(*, source_root: Path, stage_root: Path, force_restage: bool) -> Path:
    """Copy the OASIS source tree into local runtime storage for faster training I/O."""

    resolved_source = _resolve_existing_oasis_root(source_root)
    resolved_stage_root = stage_root.expanduser().resolve()
    if resolved_stage_root.exists() and force_restage:
        shutil.rmtree(resolved_stage_root)
    if resolved_stage_root.exists():
        print(f"Reusing staged OASIS source: {resolved_stage_root}")
        return resolved_stage_root

    resolved_stage_root.parent.mkdir(parents=True, exist_ok=True)
    print(f"Staging OASIS source to local disk: {resolved_source} -> {resolved_stage_root}")
    shutil.copytree(resolved_source, resolved_stage_root)
    return resolved_stage_root


def _validate_oasis_manifest(manifest_path: Path) -> int:
    """Validate that every manifest image path exists and Analyze pairs are complete."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"OASIS manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    if frame.empty:
        raise ValueError(f"OASIS manifest is empty: {manifest_path}")

    missing_paths: list[str] = []
    missing_pairs: list[str] = []
    for image_value in frame["image"].tolist():
        image_path = Path(str(image_value))
        if not image_path.exists():
            missing_paths.append(str(image_path))
            continue
        if image_path.suffix.lower() == ".hdr":
            paired_img = image_path.with_suffix(".img")
            if not paired_img.exists():
                missing_pairs.append(str(paired_img))

    if missing_paths or missing_pairs:
        raise FileNotFoundError(
            "OASIS manifest validation failed. "
            f"missing_paths={missing_paths[:10]} missing_analyze_pairs={missing_pairs[:10]}"
        )
    return int(len(frame))


def _build_manifest() -> tuple[Path, Path, int]:
    """Build the OASIS-1 manifest inside the configured runtime data root."""

    from src.configs.runtime import get_app_settings
    from src.data.oasis1 import build_oasis1_manifest

    get_app_settings.cache_clear()
    settings = get_app_settings()
    result = build_oasis1_manifest(output_format="csv")
    if result.manifest_csv_path is None:
        raise FileNotFoundError("OASIS manifest build completed without producing a CSV manifest.")

    manifest_path = result.manifest_csv_path
    summary_path = settings.data_root / "interim" / "oasis1_manifest_summary.json"
    row_count = _validate_oasis_manifest(manifest_path)
    return manifest_path, summary_path, row_count


def _signature_without_evaluations(signature: dict[str, Any]) -> dict[str, Any]:
    """Drop evaluation-only keys when comparing training-resume compatibility."""

    reduced = dict(signature)
    reduced.pop("evaluate_splits", None)
    return reduced


def _load_run_signature(run_root: Path) -> dict[str, Any] | None:
    """Load a compact training signature from persisted run metadata."""

    summary_path = run_root / "reports" / "experiment_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        config_payload = dict(payload.get("config", {}))
        training_payload = dict(config_payload.get("training", {}))
        data_payload = dict(training_payload.get("data", {}))
        early_stopping_payload = dict(training_payload.get("early_stopping", {}))
        return {
            "epochs": training_payload.get("epochs"),
            "batch_size": data_payload.get("batch_size"),
            "gradient_accumulation_steps": data_payload.get("gradient_accumulation_steps", 1),
            "image_size": tuple(data_payload.get("image_size", ())),
            "seed": data_payload.get("seed"),
            "split_seed": data_payload.get("split_seed"),
            "weighted_sampling": data_payload.get("weighted_sampling"),
            "early_stopping_monitor": early_stopping_payload.get("monitor"),
            "early_stopping_mode": early_stopping_payload.get("mode"),
            "early_stopping_patience": early_stopping_payload.get("patience"),
            "evaluate_splits": tuple(config_payload.get("evaluate_splits", ())),
        }

    resolved_config_path = run_root / "configs" / "resolved_config.json"
    if resolved_config_path.exists():
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        training_payload = dict(payload.get("training", {}))
        data_payload = dict(training_payload.get("data", {}))
        early_stopping_payload = dict(training_payload.get("early_stopping", {}))
        return {
            "epochs": training_payload.get("epochs"),
            "batch_size": data_payload.get("batch_size"),
            "gradient_accumulation_steps": data_payload.get("gradient_accumulation_steps", 1),
            "image_size": tuple(data_payload.get("image_size", ())),
            "seed": data_payload.get("seed"),
            "split_seed": data_payload.get("split_seed"),
            "weighted_sampling": data_payload.get("weighted_sampling"),
            "early_stopping_monitor": early_stopping_payload.get("monitor"),
            "early_stopping_mode": early_stopping_payload.get("mode"),
            "early_stopping_patience": early_stopping_payload.get("patience"),
        }
    return None


def _maybe_resume_checkpoint(
    *,
    outputs_root: Path,
    run_name: str,
    enabled: bool,
    requested_signature: dict[str, Any] | None = None,
) -> Path | None:
    """Return the last checkpoint path when auto-resume is enabled and config-compatible."""

    if not enabled:
        return None
    run_root = outputs_root / "runs" / "oasis" / run_name
    checkpoint_path = run_root / "checkpoints" / "last_model.pt"
    if checkpoint_path.exists():
        if requested_signature is not None:
            existing_signature = _load_run_signature(run_root)
            if existing_signature is not None and _signature_without_evaluations(existing_signature) != _signature_without_evaluations(requested_signature):
                raise ValueError(
                    f"Existing run {run_name!r} has a different training signature at {run_root}. "
                    "Use a new run name or remove the old run before retraining."
                )
        return checkpoint_path
    return None


def _evaluation_by_split(evaluations: list[Any]) -> dict[str, Any]:
    """Index evaluation results by split name."""

    return {evaluation.config.split: evaluation for evaluation in evaluations}


def _maybe_reuse_completed_run(
    *,
    outputs_root: Path,
    run_name: str,
    evaluate_splits: tuple[str, ...],
    requested_signature: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Reuse a completed persisted run when its training/evaluation artifacts already exist."""

    run_root = outputs_root / "runs" / "oasis" / run_name
    summary_path = run_root / "reports" / "experiment_summary.json"
    if not summary_path.exists():
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    if requested_signature is not None:
        config_payload = dict(payload.get("config", {}))
        training_payload = dict(config_payload.get("training", {}))
        data_payload = dict(training_payload.get("data", {}))
        early_stopping_payload = dict(training_payload.get("early_stopping", {}))
        existing_signature = {
            "epochs": training_payload.get("epochs"),
            "batch_size": data_payload.get("batch_size"),
            "gradient_accumulation_steps": data_payload.get("gradient_accumulation_steps", 1),
            "image_size": tuple(data_payload.get("image_size", ())),
            "seed": data_payload.get("seed"),
            "split_seed": data_payload.get("split_seed"),
            "weighted_sampling": data_payload.get("weighted_sampling"),
            "early_stopping_monitor": early_stopping_payload.get("monitor"),
            "early_stopping_mode": early_stopping_payload.get("mode"),
            "early_stopping_patience": early_stopping_payload.get("patience"),
            "evaluate_splits": tuple(config_payload.get("evaluate_splits", ())),
        }
        if existing_signature != requested_signature:
            return None

    training = dict(payload.get("training", {}))
    evaluations_payload = dict(payload.get("evaluations", {}))
    missing_splits = [split for split in evaluate_splits if split not in evaluations_payload]
    if missing_splits:
        return None

    best_checkpoint = Path(str(training.get("best_checkpoint", run_root / "checkpoints" / "best_model.pt")))
    last_checkpoint = Path(str(training.get("last_checkpoint", run_root / "checkpoints" / "last_model.pt")))
    evaluation_state: dict[str, dict[str, Any]] = {}
    for split in evaluate_splits:
        split_payload = dict(evaluations_payload.get(split, {}))
        metrics_json_path = Path(str(split_payload.get("metrics_json_path", run_root / "evaluation" / f"post_train_{split}_best_model" / "metrics.json")))
        predictions_csv_path = Path(str(split_payload.get("predictions_csv_path", run_root / "evaluation" / f"post_train_{split}_best_model" / "predictions.csv")))
        if not metrics_json_path.exists() or not predictions_csv_path.exists():
            return None
        evaluation_state[split] = {
            "metrics": dict(split_payload.get("metrics", {})),
            "metrics_json_path": metrics_json_path,
            "predictions_csv_path": predictions_csv_path,
        }

    return {
        "run_root": run_root,
        "best_checkpoint": best_checkpoint if best_checkpoint.exists() else None,
        "last_checkpoint": last_checkpoint if last_checkpoint.exists() else None,
        "epoch_metrics_csv_path": run_root / "metrics" / "epoch_metrics.csv",
        "summary_json_path": summary_path,
        "evaluations": evaluation_state,
        "training": training,
    }


def _write_run_summary(*, summary_path: Path, payload: dict[str, Any]) -> None:
    """Persist a compact Colab-oriented summary JSON next to the run artifacts."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Create the Colab CLI parser."""

    parser = argparse.ArgumentParser(description="Run the OASIS Colab pipeline with persistent runtime roots.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="oasis_baseline_colab_gpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--oasis-source-dir", type=Path, required=True)
    parser.add_argument("--kaggle-source-dir", type=Path, default=None)
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--image-size", nargs=3, type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--weighted-sampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--early-stopping-monitor", type=str, default=None)
    parser.add_argument("--early-stopping-mode", choices=("min", "max"), default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--evaluate-splits", nargs="+", choices=["val", "test"], default=("val", "test"))
    parser.add_argument("--stage-oasis-to-local", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-oasis-root", type=Path, default=Path("/content/oasis_stage/OASIS"))
    parser.add_argument("--force-restage", action="store_true")
    parser.add_argument("--resume-if-available", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibrate-threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--selection-metric", type=str, default="f1")
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--promote", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    """Run the Colab pipeline."""

    args = build_parser().parse_args()
    _mount_drive_if_requested(args.mount_drive)

    project_root = args.project_root.expanduser().resolve() if args.project_root is not None else Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    resolved_oasis_source = _resolve_existing_oasis_root(args.oasis_source_dir)
    resolved_training_oasis_source = (
        _stage_oasis_source(
            source_root=resolved_oasis_source,
            stage_root=args.local_oasis_root,
            force_restage=args.force_restage,
        )
        if args.stage_oasis_to_local
        else resolved_oasis_source
    )

    data_root, outputs_root = _configure_runtime_roots(project_root=project_root, runtime_root=args.runtime_root)
    os.environ["ALZ_OASIS_SOURCE_DIR"] = str(resolved_training_oasis_source)
    if args.kaggle_source_dir is not None:
        os.environ["ALZ_KAGGLE_SOURCE_DIR"] = str(args.kaggle_source_dir.expanduser().resolve())

    from src.configs.runtime import get_app_settings
    from src.evaluation.thresholds import calibrate_binary_threshold
    from src.models.registry import promote_oasis_checkpoint
    from src.training.oasis_experiment import OASISExperimentConfig, run_oasis_experiment
    from src.training.oasis_research import (
        CheckpointConfig,
        EarlyStoppingConfig,
        ResearchDataConfig,
        load_research_oasis_training_config,
    )

    get_app_settings.cache_clear()
    manifest_path, summary_path, manifest_rows = _build_manifest()

    config_path = args.config or (project_root / "configs" / "oasis_train_colab_gpu.yaml")
    training_cfg = load_research_oasis_training_config(config_path)
    resolved_device, enable_mixed_precision = _resolve_training_device(args.device)
    resolved_num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else (0 if _running_in_colab() else training_cfg.data.num_workers)
    )
    data_cfg = ResearchDataConfig(
        batch_size=int(args.batch_size) if args.batch_size is not None else training_cfg.data.batch_size,
        gradient_accumulation_steps=(
            int(args.gradient_accumulation_steps)
            if args.gradient_accumulation_steps is not None
            else training_cfg.data.gradient_accumulation_steps
        ),
        num_workers=resolved_num_workers,
        cache_rate=float(args.cache_rate) if args.cache_rate is not None else training_cfg.data.cache_rate,
        image_size=tuple(args.image_size) if args.image_size is not None else training_cfg.data.image_size,
        seed=int(args.seed) if args.seed is not None else training_cfg.data.seed,
        split_seed=int(args.split_seed) if args.split_seed is not None else training_cfg.data.split_seed,
        train_fraction=training_cfg.data.train_fraction,
        val_fraction=training_cfg.data.val_fraction,
        test_fraction=training_cfg.data.test_fraction,
        weighted_sampling=(
            bool(args.weighted_sampling)
            if args.weighted_sampling is not None
            else training_cfg.data.weighted_sampling
        ),
        max_train_batches=training_cfg.data.max_train_batches,
        max_val_batches=training_cfg.data.max_val_batches,
    )
    early_stopping_cfg = EarlyStoppingConfig(
        enabled=training_cfg.early_stopping.enabled,
        patience=(
            int(args.early_stopping_patience)
            if args.early_stopping_patience is not None
            else training_cfg.early_stopping.patience
        ),
        min_delta=(
            float(args.early_stopping_min_delta)
            if args.early_stopping_min_delta is not None
            else training_cfg.early_stopping.min_delta
        ),
        monitor=args.early_stopping_monitor or training_cfg.early_stopping.monitor,
        mode=args.early_stopping_mode or training_cfg.early_stopping.mode,
    )
    training_cfg = replace(
        training_cfg,
        epochs=int(args.epochs) if args.epochs is not None else training_cfg.epochs,
        run_name=args.run_name,
        device=resolved_device,
        mixed_precision=enable_mixed_precision,
        data=data_cfg,
        early_stopping=early_stopping_cfg,
    )
    requested_signature = {
        "epochs": training_cfg.epochs,
        "batch_size": training_cfg.data.batch_size,
        "gradient_accumulation_steps": training_cfg.data.gradient_accumulation_steps,
        "image_size": tuple(training_cfg.data.image_size),
        "seed": training_cfg.data.seed,
        "split_seed": training_cfg.data.split_seed,
        "weighted_sampling": training_cfg.data.weighted_sampling,
        "early_stopping_monitor": training_cfg.early_stopping.monitor,
        "early_stopping_mode": training_cfg.early_stopping.mode,
        "early_stopping_patience": training_cfg.early_stopping.patience,
        "evaluate_splits": tuple(args.evaluate_splits),
    }
    resume_checkpoint = _maybe_resume_checkpoint(
        outputs_root=outputs_root,
        run_name=args.run_name,
        enabled=bool(args.resume_if_available),
        requested_signature=requested_signature,
    )
    checkpoint_cfg = replace(training_cfg.checkpoint, resume_from=resume_checkpoint)
    training_cfg = replace(training_cfg, checkpoint=checkpoint_cfg)

    print(f"project_root={project_root}")
    print(f"data_root={data_root}")
    print(f"outputs_root={outputs_root}")
    print(f"oasis_source_for_training={resolved_training_oasis_source}")
    print(f"manifest_path={manifest_path}")
    print(f"manifest_summary_path={summary_path}")
    print(f"manifest_rows={manifest_rows}")
    print(f"using_device={training_cfg.device} mixed_precision={training_cfg.mixed_precision}")
    print(
        f"batch_size={training_cfg.data.batch_size} grad_accum={training_cfg.data.gradient_accumulation_steps} "
        f"num_workers={training_cfg.data.num_workers} weighted_sampling={training_cfg.data.weighted_sampling}"
    )
    print(
        f"early_stopping={training_cfg.early_stopping.monitor}/{training_cfg.early_stopping.mode} "
        f"patience={training_cfg.early_stopping.patience}"
    )
    if resume_checkpoint is not None:
        print(f"resume_checkpoint={resume_checkpoint}")
    else:
        print("resume_checkpoint=None")

    existing_run = (
        _maybe_reuse_completed_run(
            outputs_root=outputs_root,
            run_name=args.run_name,
            evaluate_splits=tuple(args.evaluate_splits),
            requested_signature=requested_signature,
        )
        if args.resume_if_available
        else None
    )

    if existing_run is not None:
        print(f"Reusing completed persisted run {args.run_name} from {existing_run['run_root']}")
        resolved_run_name = args.run_name
        run_root = Path(existing_run["run_root"])
        best_checkpoint = existing_run["best_checkpoint"]
        last_checkpoint = existing_run["last_checkpoint"]
        epoch_metrics_csv_path = Path(existing_run["epoch_metrics_csv_path"])
        evaluations_by_split = dict(existing_run["evaluations"])
    else:
        result = run_oasis_experiment(
            OASISExperimentConfig(
                training=training_cfg,
                evaluate_splits=tuple(args.evaluate_splits),
                checkpoint_name="best_model.pt",
            )
        )
        resolved_run_name = result.training.run_name
        run_root = Path(result.training.run_root)
        best_checkpoint = result.training.best_checkpoint_path
        last_checkpoint = result.training.last_checkpoint_path
        epoch_metrics_csv_path = result.training.epoch_metrics_csv_path
        evaluations_by_split = {
            split: {
                "metrics": dict(evaluation.result.metrics),
                "metrics_json_path": Path(evaluation.paths.metrics_json_path),
                "predictions_csv_path": Path(evaluation.paths.predictions_csv_path),
            }
            for split, evaluation in _evaluation_by_split(result.evaluations).items()
        }

    print(f"run_name={resolved_run_name}")
    print(f"run_root={run_root}")
    print(f"best_checkpoint={best_checkpoint}")
    print(f"last_checkpoint={last_checkpoint}")
    print(f"epoch_metrics_csv={epoch_metrics_csv_path}")
    for split, evaluation in evaluations_by_split.items():
        print(f"{split}_metrics_json={evaluation['metrics_json_path']}")
        print(f"{split}_predictions_csv={evaluation['predictions_csv_path']}")
        print(f"{split}_accuracy={evaluation['metrics'].get('accuracy')}")
        print(f"{split}_auroc={evaluation['metrics'].get('auroc')}")

    calibration_result = None
    if args.calibrate_threshold:
        if "val" not in evaluations_by_split:
            raise ValueError("Threshold calibration requires a validation evaluation. Include `val` in --evaluate-splits.")
        calibration_dir = run_root / "calibration" / f"threshold_{args.selection_metric}"
        calibration_result = calibrate_binary_threshold(
            validation_predictions_path=evaluations_by_split["val"]["predictions_csv_path"],
            test_predictions_path=(
                evaluations_by_split["test"]["predictions_csv_path"]
                if "test" in evaluations_by_split
                else None
            ),
            output_dir=calibration_dir,
            selection_metric=args.selection_metric,
            threshold_step=args.threshold_step,
        )
        print(f"recommended_threshold={calibration_result.threshold}")
        print(f"threshold_calibration_path={calibration_result.calibration_report_path}")

    registry_entry = None
    registry_path = None
    if args.promote:
        if best_checkpoint is None:
            raise FileNotFoundError("Promotion requested, but the training run did not produce a best checkpoint.")
        registry_entry, registry_path = promote_oasis_checkpoint(
            run_name=resolved_run_name,
            checkpoint_path=best_checkpoint,
            val_metrics_path=(
                evaluations_by_split["val"]["metrics_json_path"] if "val" in evaluations_by_split else None
            ),
            test_metrics_path=(
                evaluations_by_split["test"]["metrics_json_path"] if "test" in evaluations_by_split else None
            ),
            model_config_path=project_root / "configs" / "oasis_model.yaml",
            preprocessing_config_path=project_root / "configs" / "oasis_transforms.yaml",
            image_size=tuple(training_cfg.data.image_size),
            threshold_calibration_path=(
                calibration_result.calibration_report_path if calibration_result is not None else None
            ),
        )
        print(f"registry_path={registry_path}")
        print(f"registry_recommended_threshold={registry_entry.recommended_threshold}")

    summary = {
        "run_name": resolved_run_name,
        "run_root": str(run_root),
        "best_checkpoint": None if best_checkpoint is None else str(best_checkpoint),
        "last_checkpoint": None if last_checkpoint is None else str(last_checkpoint),
        "manifest_path": str(manifest_path),
        "manifest_summary_path": str(summary_path),
        "manifest_rows": manifest_rows,
        "data_root": str(data_root),
        "outputs_root": str(outputs_root),
        "oasis_source_for_training": str(resolved_training_oasis_source),
        "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
        "threshold_calibration_path": None
        if calibration_result is None
        else str(calibration_result.calibration_report_path),
        "registry_path": None if registry_path is None else str(registry_path),
        "evaluations": {
            split: {
                "accuracy": evaluation["metrics"].get("accuracy"),
                "auroc": evaluation["metrics"].get("auroc"),
                "metrics_json_path": str(evaluation["metrics_json_path"]),
                "predictions_csv_path": str(evaluation["predictions_csv_path"]),
            }
            for split, evaluation in evaluations_by_split.items()
        },
    }
    summary_path_out = run_root / "reports" / "colab_run_summary.json"
    _write_run_summary(summary_path=summary_path_out, payload=summary)
    print(f"colab_run_summary={summary_path_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
