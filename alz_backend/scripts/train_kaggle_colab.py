"""Colab-friendly Kaggle training pipeline with persistent runtime roots."""

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


def _looks_like_kaggle_root(candidate: Path) -> bool:
    """Return whether a folder looks like a Kaggle upload bundle root."""

    return (candidate / "OriginalDataset").is_dir() or (candidate / "AugmentedAlzheimerDataset").is_dir()


def _resolve_existing_kaggle_root(source_root: Path) -> Path:
    """Return the concrete Kaggle bundle root, tolerating one nested folder."""

    resolved = source_root.expanduser().resolve()
    if _looks_like_kaggle_root(resolved):
        return resolved

    nested_candidates = [child for child in resolved.iterdir() if child.is_dir()] if resolved.exists() else []
    matching_children = [child for child in nested_candidates if _looks_like_kaggle_root(child)]
    if len(matching_children) == 1:
        return matching_children[0].resolve()

    raise FileNotFoundError(
        "Could not resolve a Kaggle bundle root containing `OriginalDataset/` or "
        f"`AugmentedAlzheimerDataset/` beneath {resolved}"
    )


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


def _stage_kaggle_source(*, source_root: Path, stage_root: Path, force_restage: bool) -> Path:
    """Copy the Kaggle source tree into local runtime storage for faster training I/O."""

    resolved_source = _resolve_existing_kaggle_root(source_root)
    resolved_stage_root = stage_root.expanduser().resolve()
    if resolved_stage_root.exists() and force_restage:
        shutil.rmtree(resolved_stage_root)
    if resolved_stage_root.exists():
        print(f"Reusing staged Kaggle source: {resolved_stage_root}")
        return resolved_stage_root

    resolved_stage_root.parent.mkdir(parents=True, exist_ok=True)
    print(f"Staging Kaggle source to local disk: {resolved_source} -> {resolved_stage_root}")
    shutil.copytree(resolved_source, resolved_stage_root)
    return resolved_stage_root


def _validate_kaggle_manifest(manifest_path: Path) -> int:
    """Validate that every manifest image path exists."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Kaggle manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    if frame.empty:
        raise ValueError(f"Kaggle manifest is empty: {manifest_path}")

    missing_paths: list[str] = []
    for image_value in frame["image"].tolist():
        image_path = Path(str(image_value))
        if not image_path.exists():
            missing_paths.append(str(image_path))

    if missing_paths:
        raise FileNotFoundError(
            "Kaggle manifest validation failed. "
            f"missing_paths={missing_paths[:10]}"
        )
    return int(len(frame))


def _build_manifest_and_splits(
    *,
    split_random_state: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, Any]:
    """Build the Kaggle manifest and leakage-aware splits inside the runtime data root."""

    from src.configs.runtime import get_app_settings
    from src.data.kaggle_alz import build_kaggle_manifest
    from src.data.kaggle_splits import build_kaggle_splits

    get_app_settings.cache_clear()
    settings = get_app_settings()
    manifest_result = build_kaggle_manifest(settings=settings, output_format="csv")
    if manifest_result.manifest_csv_path is None:
        raise FileNotFoundError("Kaggle manifest build completed without producing a CSV manifest.")

    manifest_path = manifest_result.manifest_csv_path
    manifest_rows = _validate_kaggle_manifest(manifest_path)
    split_result = build_kaggle_splits(
        settings=settings,
        manifest_path=manifest_path,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_state=split_random_state,
    )

    split_counts = {
        "train_rows": split_result.train_rows,
        "val_rows": split_result.val_rows,
        "test_rows": split_result.test_rows,
    }
    if any(count <= 0 for count in split_counts.values()):
        raise ValueError(f"Kaggle split validation failed with non-positive counts: {split_counts}")

    return {
        "manifest_path": manifest_path,
        "manifest_summary_path": manifest_result.summary_path,
        "manifest_rows": manifest_rows,
        "split_assignments_path": split_result.split_assignments_path,
        "train_manifest_path": split_result.train_manifest_path,
        "val_manifest_path": split_result.val_manifest_path,
        "test_manifest_path": split_result.test_manifest_path,
        "split_summary_path": split_result.summary_path,
        "train_rows": split_result.train_rows,
        "val_rows": split_result.val_rows,
        "test_rows": split_result.test_rows,
    }


def _signature_matches(existing_signature: dict[str, Any] | None, requested_signature: dict[str, Any]) -> bool:
    """Return whether an existing signature is compatible with the requested one."""

    if not existing_signature:
        return True
    for key, existing_value in existing_signature.items():
        if key not in requested_signature:
            continue
        normalized_existing = tuple(existing_value) if isinstance(existing_value, list) else existing_value
        normalized_requested = (
            tuple(requested_signature[key])
            if isinstance(requested_signature[key], list)
            else requested_signature[key]
        )
        if normalized_requested != normalized_existing:
            return False
    return True


def _load_run_signature(run_root: Path) -> dict[str, Any] | None:
    """Load a compact training signature from persisted run metadata."""

    requested_signature_path = run_root / "reports" / "colab_requested_signature.json"
    if requested_signature_path.exists():
        payload = json.loads(requested_signature_path.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else None

    summary_path = run_root / "reports" / "colab_run_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        signature = payload.get("signature")
        return dict(signature) if isinstance(signature, dict) else None

    resolved_config_path = run_root / "configs" / "resolved_config.json"
    if resolved_config_path.exists():
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        training_payload = dict(payload.get("training", {}))
        data_payload = dict(training_payload.get("data", {}))
        early_stopping_payload = dict(training_payload.get("early_stopping", {}))
        return {
            "epochs": training_payload.get("epochs"),
            "dry_run": training_payload.get("dry_run"),
            "dropout_prob": training_payload.get("dropout_prob"),
            "batch_size": data_payload.get("batch_size"),
            "gradient_accumulation_steps": data_payload.get("gradient_accumulation_steps", 1),
            "image_size_2d": tuple(data_payload.get("image_size_2d", ())),
            "image_size_3d": tuple(data_payload.get("image_size_3d", ())),
            "seed": data_payload.get("seed"),
            "early_stopping_monitor": early_stopping_payload.get("monitor"),
            "early_stopping_mode": early_stopping_payload.get("mode"),
            "early_stopping_patience": early_stopping_payload.get("patience"),
        }
    return None


def _persist_requested_signature(*, outputs_root: Path, run_name: str, requested_signature: dict[str, Any]) -> Path:
    """Persist the requested signature before training so resume checks survive restarts."""

    signature_path = outputs_root / "runs" / "kaggle" / run_name / "reports" / "colab_requested_signature.json"
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(json.dumps(requested_signature, indent=2), encoding="utf-8")
    return signature_path


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
    run_root = outputs_root / "runs" / "kaggle" / run_name
    checkpoint_path = run_root / "checkpoints" / "last_model.pt"
    if checkpoint_path.exists():
        if requested_signature is not None:
            existing_signature = _load_run_signature(run_root)
            if not _signature_matches(existing_signature, requested_signature):
                raise ValueError(
                    f"Existing run {run_name!r} has a different training signature at {run_root}. "
                    "Use a new run name or remove the old run before retraining."
                )
        return checkpoint_path
    return None


def _maybe_reuse_completed_run(
    *,
    outputs_root: Path,
    run_name: str,
    requested_signature: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Reuse a completed persisted run when its training artifacts already exist."""

    run_root = outputs_root / "runs" / "kaggle" / run_name
    final_metrics_path = run_root / "metrics" / "final_metrics.json"
    resolved_config_path = run_root / "configs" / "resolved_config.json"
    if not final_metrics_path.exists() or not resolved_config_path.exists():
        return None

    if requested_signature is not None:
        existing_signature = _load_run_signature(run_root)
        if not _signature_matches(existing_signature, requested_signature):
            raise ValueError(
                f"Existing completed run {run_name!r} has a different training signature at {run_root}. "
                "Use a new run name or remove the old run before retraining."
            )

    best_checkpoint = run_root / "checkpoints" / "best_model.pt"
    last_checkpoint = run_root / "checkpoints" / "last_model.pt"
    epoch_metrics_csv_path = run_root / "metrics" / "epoch_metrics.csv"
    val_predictions_path = run_root / "evaluation" / "val_predictions.csv"
    test_predictions_path = run_root / "evaluation" / "test_predictions.csv"

    if not best_checkpoint.exists() and not last_checkpoint.exists():
        return None
    if not epoch_metrics_csv_path.exists():
        return None

    final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
    if not isinstance(final_metrics, dict):
        return None

    return {
        "run_root": run_root,
        "best_checkpoint": best_checkpoint if best_checkpoint.exists() else None,
        "last_checkpoint": last_checkpoint if last_checkpoint.exists() else None,
        "epoch_metrics_csv_path": epoch_metrics_csv_path,
        "final_metrics_path": final_metrics_path,
        "resolved_config_path": resolved_config_path,
        "summary_report_path": run_root / "reports" / "summary_report.md",
        "val_predictions_path": val_predictions_path if val_predictions_path.exists() else None,
        "test_predictions_path": test_predictions_path if test_predictions_path.exists() else None,
        "final_metrics": final_metrics,
    }


def _write_run_summary(*, summary_path: Path, payload: dict[str, Any]) -> None:
    """Persist a compact Colab-oriented summary JSON next to the run artifacts."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Create the Colab CLI parser."""

    parser = argparse.ArgumentParser(description="Run the Kaggle Colab pipeline with persistent runtime roots.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="kaggle_colab_baseline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--kaggle-source-dir", type=Path, required=True)
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dropout-prob", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-rate", type=float, default=None)
    parser.add_argument("--image-size-2d", nargs=2, type=int, default=None)
    parser.add_argument("--image-size-3d", nargs=3, type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--split-random-state", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--early-stopping-monitor", type=str, default=None)
    parser.add_argument("--early-stopping-mode", choices=("min", "max"), default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--stage-kaggle-to-local", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--local-kaggle-root",
        type=Path,
        default=Path("/content/kaggle_stage/kaggle_alz_upload_bundle"),
    )
    parser.add_argument("--force-restage", action="store_true")
    parser.add_argument("--resume-if-available", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    """Run the Colab pipeline."""

    args = build_parser().parse_args()
    _mount_drive_if_requested(args.mount_drive)

    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    resolved_kaggle_source = _resolve_existing_kaggle_root(args.kaggle_source_dir)
    resolved_training_kaggle_source = (
        _stage_kaggle_source(
            source_root=resolved_kaggle_source,
            stage_root=args.local_kaggle_root,
            force_restage=args.force_restage,
        )
        if args.stage_kaggle_to_local
        else resolved_kaggle_source
    )

    data_root, outputs_root = _configure_runtime_roots(project_root=project_root, runtime_root=args.runtime_root)
    os.environ["ALZ_KAGGLE_SOURCE_DIR"] = str(resolved_training_kaggle_source)

    from src.configs.runtime import get_app_settings
    from src.training.kaggle_research import (
        CheckpointConfig,
        EarlyStoppingConfig,
        ResearchKaggleDataConfig,
        load_research_kaggle_training_config,
        run_research_kaggle_training,
    )

    get_app_settings.cache_clear()
    manifest_state = _build_manifest_and_splits(
        split_random_state=int(args.split_random_state),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
    )

    config_path = args.config or (project_root / "configs" / "kaggle_train.yaml")
    training_cfg = load_research_kaggle_training_config(config_path)
    resolved_device, enable_cuda_mixed_precision = _resolve_training_device(args.device)
    resolved_num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else (0 if _running_in_colab() else training_cfg.data.num_workers)
    )
    requested_mixed_precision = (
        bool(args.mixed_precision)
        if args.mixed_precision is not None
        else training_cfg.mixed_precision
    )
    data_cfg = ResearchKaggleDataConfig(
        batch_size=int(args.batch_size) if args.batch_size is not None else training_cfg.data.batch_size,
        gradient_accumulation_steps=(
            int(args.gradient_accumulation_steps)
            if args.gradient_accumulation_steps is not None
            else training_cfg.data.gradient_accumulation_steps
        ),
        num_workers=resolved_num_workers,
        cache_rate=float(args.cache_rate) if args.cache_rate is not None else training_cfg.data.cache_rate,
        image_size_2d=tuple(args.image_size_2d) if args.image_size_2d is not None else training_cfg.data.image_size_2d,
        image_size_3d=tuple(args.image_size_3d) if args.image_size_3d is not None else training_cfg.data.image_size_3d,
        seed=int(args.seed) if args.seed is not None else training_cfg.data.seed,
        label_map=training_cfg.data.label_map,
        max_train_batches=args.max_train_batches if args.max_train_batches is not None else training_cfg.data.max_train_batches,
        max_val_batches=args.max_val_batches if args.max_val_batches is not None else training_cfg.data.max_val_batches,
        max_test_batches=args.max_test_batches if args.max_test_batches is not None else training_cfg.data.max_test_batches,
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
    provisional_cfg = replace(
        training_cfg,
        run_name=args.run_name,
        epochs=int(args.epochs) if args.epochs is not None else training_cfg.epochs,
        device=resolved_device,
        mixed_precision=bool(requested_mixed_precision and enable_cuda_mixed_precision),
        deterministic=bool(args.deterministic) if args.deterministic is not None else training_cfg.deterministic,
        dry_run=bool(args.dry_run) if args.dry_run is not None else training_cfg.dry_run,
        dropout_prob=float(args.dropout_prob) if args.dropout_prob is not None else training_cfg.dropout_prob,
        data=data_cfg,
        early_stopping=early_stopping_cfg,
    )
    requested_signature = {
        "epochs": provisional_cfg.epochs,
        "dry_run": provisional_cfg.dry_run,
        "dropout_prob": provisional_cfg.dropout_prob,
        "batch_size": provisional_cfg.data.batch_size,
        "gradient_accumulation_steps": provisional_cfg.data.gradient_accumulation_steps,
        "image_size_2d": tuple(provisional_cfg.data.image_size_2d),
        "image_size_3d": tuple(provisional_cfg.data.image_size_3d),
        "seed": provisional_cfg.data.seed,
        "split_random_state": int(args.split_random_state),
        "train_fraction": float(args.train_fraction),
        "val_fraction": float(args.val_fraction),
        "test_fraction": float(args.test_fraction),
        "early_stopping_monitor": provisional_cfg.early_stopping.monitor,
        "early_stopping_mode": provisional_cfg.early_stopping.mode,
        "early_stopping_patience": provisional_cfg.early_stopping.patience,
    }
    resume_checkpoint = _maybe_resume_checkpoint(
        outputs_root=outputs_root,
        run_name=args.run_name,
        enabled=bool(args.resume_if_available),
        requested_signature=requested_signature,
    )
    checkpoint_cfg = replace(training_cfg.checkpoint, resume_from=resume_checkpoint)
    training_cfg = replace(provisional_cfg, checkpoint=checkpoint_cfg)
    signature_path = outputs_root / "runs" / "kaggle" / args.run_name / "reports" / "colab_requested_signature.json"

    print(f"project_root={project_root}")
    print(f"data_root={data_root}")
    print(f"outputs_root={outputs_root}")
    print(f"kaggle_source_for_training={resolved_training_kaggle_source}")
    print(f"manifest_path={manifest_state['manifest_path']}")
    print(f"manifest_summary_path={manifest_state['manifest_summary_path']}")
    print(f"manifest_rows={manifest_state['manifest_rows']}")
    print(f"split_summary_path={manifest_state['split_summary_path']}")
    print(
        "split_rows="
        f"train:{manifest_state['train_rows']} "
        f"val:{manifest_state['val_rows']} "
        f"test:{manifest_state['test_rows']}"
    )
    print(f"colab_requested_signature={signature_path}")
    print(f"using_device={training_cfg.device} mixed_precision={training_cfg.mixed_precision}")
    print(
        f"batch_size={training_cfg.data.batch_size} grad_accum={training_cfg.data.gradient_accumulation_steps} "
        f"num_workers={training_cfg.data.num_workers} dry_run={training_cfg.dry_run}"
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
        final_metrics_path = Path(existing_run["final_metrics_path"])
        resolved_config_path = Path(existing_run["resolved_config_path"])
        summary_report_path = Path(existing_run["summary_report_path"])
        final_metrics = dict(existing_run["final_metrics"])
        val_predictions_path = existing_run["val_predictions_path"]
        test_predictions_path = existing_run["test_predictions_path"]
    else:
        signature_path = _persist_requested_signature(
            outputs_root=outputs_root,
            run_name=args.run_name,
            requested_signature=requested_signature,
        )
        result = run_research_kaggle_training(training_cfg)
        resolved_run_name = result.run_name
        run_root = Path(result.run_root)
        best_checkpoint = result.best_checkpoint_path
        last_checkpoint = result.last_checkpoint_path
        epoch_metrics_csv_path = result.epoch_metrics_csv_path
        final_metrics_path = result.final_metrics_path
        resolved_config_path = result.resolved_config_path
        summary_report_path = result.summary_report_path
        final_metrics = dict(result.final_metrics)
        val_predictions_path = result.val_predictions_path
        test_predictions_path = result.test_predictions_path

    print(f"run_name={resolved_run_name}")
    print(f"run_root={run_root}")
    print(f"best_checkpoint={best_checkpoint}")
    print(f"last_checkpoint={last_checkpoint}")
    print(f"epoch_metrics_csv={epoch_metrics_csv_path}")
    print(f"final_metrics={final_metrics_path}")
    print(f"resolved_config={resolved_config_path}")
    print(f"summary_report={summary_report_path}")
    if val_predictions_path is not None:
        print(f"val_predictions={val_predictions_path}")
    if test_predictions_path is not None:
        print(f"test_predictions={test_predictions_path}")
    print(f"val_accuracy={final_metrics.get('validation', {}).get('accuracy')}")
    print(f"val_macro_f1={final_metrics.get('validation', {}).get('macro_f1')}")
    print(f"val_macro_ovr_auroc={final_metrics.get('validation', {}).get('macro_ovr_auroc')}")
    print(f"test_accuracy={final_metrics.get('test', {}).get('accuracy')}")
    print(f"test_macro_f1={final_metrics.get('test', {}).get('macro_f1')}")
    print(f"test_macro_ovr_auroc={final_metrics.get('test', {}).get('macro_ovr_auroc')}")

    summary = {
        "run_name": resolved_run_name,
        "run_root": str(run_root),
        "best_checkpoint": None if best_checkpoint is None else str(best_checkpoint),
        "last_checkpoint": None if last_checkpoint is None else str(last_checkpoint),
        "manifest_path": str(manifest_state["manifest_path"]),
        "manifest_summary_path": str(manifest_state["manifest_summary_path"]),
        "manifest_rows": manifest_state["manifest_rows"],
        "split_assignments_path": str(manifest_state["split_assignments_path"]),
        "train_manifest_path": str(manifest_state["train_manifest_path"]),
        "val_manifest_path": str(manifest_state["val_manifest_path"]),
        "test_manifest_path": str(manifest_state["test_manifest_path"]),
        "split_summary_path": str(manifest_state["split_summary_path"]),
        "train_rows": manifest_state["train_rows"],
        "val_rows": manifest_state["val_rows"],
        "test_rows": manifest_state["test_rows"],
        "data_root": str(data_root),
        "outputs_root": str(outputs_root),
        "kaggle_source_for_training": str(resolved_training_kaggle_source),
        "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
        "epoch_metrics_csv_path": str(epoch_metrics_csv_path),
        "final_metrics_path": str(final_metrics_path),
        "resolved_config_path": str(resolved_config_path),
        "summary_report_path": str(summary_report_path),
        "val_predictions_path": None if val_predictions_path is None else str(val_predictions_path),
        "test_predictions_path": None if test_predictions_path is None else str(test_predictions_path),
        "validation": dict(final_metrics.get("validation", {})),
        "test": dict(final_metrics.get("test", {})),
        "signature": requested_signature,
    }
    summary_path_out = run_root / "reports" / "colab_run_summary.json"
    _write_run_summary(summary_path=summary_path_out, payload=summary)
    print(f"colab_run_summary={summary_path_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
