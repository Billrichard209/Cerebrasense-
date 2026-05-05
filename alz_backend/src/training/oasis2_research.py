"""Research-grade MONAI training pipeline for supervised OASIS-2 experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from src.configs.runtime import AppSettings, get_app_settings
from src.data.oasis2_loaders import OASIS2LoaderConfig, build_oasis2_dataloaders
from src.data.oasis2_supervised import (
    build_oasis2_training_readiness_report,
    save_oasis2_training_readiness_report,
)
from src.models.factory import build_model, describe_model_config, load_oasis_model_config
from src.utils.io_utils import ensure_directory, resolve_project_root
from src.utils.monai_utils import load_torch_symbols
from src.utils.seed import build_seed_snapshot, set_global_seed

from .oasis_research import (
    ResearchDataConfig,
    ResearchOASISTrainingConfig,
    ResearchRunPaths,
    ResearchTrainingError,
    ResearchTrainingResult,
    _apply_dry_run_overrides,
    _autocast_context,
    _build_grad_scaler,
    _checkpoint_payload,
    _coerce_labels,
    _epoch_row,
    _initial_best_value,
    _is_improvement,
    _load_resume_checkpoint,
    _resolve_device,
    _resolve_monitor_value,
    _run_epoch,
    _save_checkpoint,
    _step_scheduler,
    _write_confusion_matrix,
    _write_epoch_metrics,
)
from .trainer_utils import build_classification_loss, build_optimizer, build_scheduler

_load_torch_symbols = load_torch_symbols


def default_oasis2_train_config_path() -> Path:
    """Return the default OASIS-2 training YAML path."""

    return resolve_project_root() / "configs" / "oasis2_train.yaml"


def build_oasis2_run_paths(settings: AppSettings, run_name: str) -> ResearchRunPaths:
    """Create and return the run folder structure under outputs/runs/oasis2."""

    run_root = ensure_directory(settings.outputs_root / "runs" / "oasis2" / run_name)
    checkpoint_root = ensure_directory(run_root / "checkpoints")
    metrics_root = ensure_directory(run_root / "metrics")
    reports_root = ensure_directory(run_root / "reports")
    config_root = ensure_directory(run_root / "configs")
    return ResearchRunPaths(
        run_root=run_root,
        checkpoint_root=checkpoint_root,
        metrics_root=metrics_root,
        reports_root=reports_root,
        config_root=config_root,
        best_checkpoint_path=checkpoint_root / "best_model.pt",
        last_checkpoint_path=checkpoint_root / "last_model.pt",
        epoch_metrics_csv_path=metrics_root / "epoch_metrics.csv",
        epoch_metrics_json_path=metrics_root / "epoch_metrics.json",
        confusion_matrix_path=metrics_root / "confusion_matrix.json",
        summary_report_path=reports_root / "summary_report.md",
        resolved_config_path=config_root / "resolved_config.json",
    )


def _save_resolved_oasis2_config(
    *,
    cfg: ResearchOASISTrainingConfig,
    model_cfg: object,
    paths: ResearchRunPaths,
    loader_cfg: OASIS2LoaderConfig,
    split_summary: dict[str, Any],
) -> None:
    """Persist resolved training, model, and split config for reproducibility."""

    payload = {
        "training": asdict(cfg),
        "model": describe_model_config(model_cfg),
        "seed": build_seed_snapshot(cfg.data.seed, deterministic=cfg.deterministic),
        "dataset": {
            "name": "oasis2",
            "task": "binary_structural_session_classification",
            "manifest_path": str(loader_cfg.manifest_path) if loader_cfg.manifest_path is not None else None,
            "split_plan_path": str(loader_cfg.split_plan_path) if loader_cfg.split_plan_path is not None else None,
            "split_reports_root": (
                str(Path(str(split_summary["artifacts"]["train_manifest_path"])).parent)
                if split_summary.get("artifacts", {}).get("train_manifest_path")
                else None
            ),
            "split_summary": split_summary,
        },
    }
    payload["training"]["model_config_path"] = str(cfg.model_config_path) if cfg.model_config_path else None
    payload["training"]["checkpoint"]["resume_from"] = (
        str(cfg.checkpoint.resume_from) if cfg.checkpoint.resume_from else None
    )
    paths.resolved_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_loaders(cfg: ResearchOASISTrainingConfig):
    """Build reproducible OASIS-2 loaders for training and validation."""

    loader_cfg = OASIS2LoaderConfig(
        seed=cfg.data.seed,
        split_seed=cfg.data.split_seed,
        train_fraction=cfg.data.train_fraction,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        cache_rate=cfg.data.cache_rate,
        weighted_sampling=cfg.data.weighted_sampling,
        transform_config=load_oasis_transform_config_for_oasis2(cfg),
    )
    dataloaders = build_oasis2_dataloaders(loader_cfg)
    return loader_cfg, dataloaders


def load_oasis_transform_config_for_oasis2(cfg: ResearchOASISTrainingConfig):
    """Reuse the existing OASIS transform recipe with the requested image size."""

    from src.transforms.oasis_transforms import OASISSpatialConfig, OASISTransformConfig, load_oasis_transform_config

    transform_cfg = load_oasis_transform_config()
    return OASISTransformConfig(
        load=transform_cfg.load,
        orientation=transform_cfg.orientation,
        spacing=transform_cfg.spacing,
        intensity=transform_cfg.intensity,
        skull_strip=transform_cfg.skull_strip,
        spatial=OASISSpatialConfig(spatial_size=cfg.data.image_size),
        augmentation=transform_cfg.augmentation,
    )


def _write_summary_report(
    *,
    cfg: ResearchOASISTrainingConfig,
    paths: ResearchRunPaths,
    rows: list[dict[str, Any]],
    best_epoch: int,
    best_monitor_value: float,
    stopped_early: bool,
    elapsed_seconds: float,
    split_summary: dict[str, Any],
) -> None:
    """Write a human-readable final OASIS-2 experiment report."""

    final_row = rows[-1]
    artifacts = split_summary.get("artifacts", {})
    lines = [
        f"# {cfg.run_name}",
        "",
        "This is a research-grade OASIS-2 MONAI training run for backend development.",
        "It is decision-support research software, not diagnosis software.",
        "",
        "## Run",
        "",
        f"- epochs_requested: {cfg.epochs}",
        f"- epochs_completed: {len(rows)}",
        f"- dry_run: {cfg.dry_run}",
        f"- device: {cfg.device}",
        f"- mixed_precision: {cfg.mixed_precision}",
        f"- gradient_accumulation_steps: {cfg.data.gradient_accumulation_steps}",
        f"- stopped_early: {stopped_early}",
        f"- best_epoch: {best_epoch}",
        f"- best_monitor_value: {best_monitor_value}",
        f"- elapsed_seconds: {round(elapsed_seconds, 2)}",
        "",
        "## Data",
        "",
        f"- train_manifest: {artifacts.get('train_manifest_path')}",
        f"- val_manifest: {artifacts.get('val_manifest_path')}",
        f"- test_manifest: {artifacts.get('test_manifest_path')}",
        f"- mixed_label_group_count: {split_summary.get('mixed_label_group_count', 0)}",
        "",
        "## Final Validation Metrics",
        "",
        f"- val_loss: {final_row['val_loss']:.6f}",
        f"- accuracy: {final_row['accuracy']:.6f}",
        f"- auroc: {final_row['auroc']:.6f}",
        f"- precision: {final_row['precision']:.6f}",
        f"- recall: {final_row['recall']:.6f}",
        f"- f1: {final_row['f1']:.6f}",
        f"- sensitivity: {final_row['sensitivity']:.6f}",
        f"- specificity: {final_row['specificity']:.6f}",
        "",
        "## Artifacts",
        "",
        f"- best_checkpoint: {paths.best_checkpoint_path}",
        f"- last_checkpoint: {paths.last_checkpoint_path}",
        f"- epoch_metrics_csv: {paths.epoch_metrics_csv_path}",
        f"- epoch_metrics_json: {paths.epoch_metrics_json_path}",
        f"- confusion_matrix: {paths.confusion_matrix_path}",
        f"- resolved_config: {paths.resolved_config_path}",
        "",
        "## Failure Notes",
        "",
        "- Tiny dry runs are pipeline checks and should not be interpreted as model quality.",
        "- AUROC is reported as 0.0 when the evaluated split contains only one class.",
        "- OASIS-2 training assumes an explicit binary label policy and subject-safe split plan.",
    ]
    paths.summary_report_path.write_text("\n".join(lines), encoding="utf-8")


def run_research_oasis2_training(
    config: ResearchOASISTrainingConfig | None = None,
    *,
    settings: AppSettings | None = None,
) -> ResearchTrainingResult:
    """Run the config-driven supervised OASIS-2 training pipeline."""

    print("oasis2_trainer: resolving settings", flush=True)
    resolved_settings = settings or get_app_settings()
    print("oasis2_trainer: building config", flush=True)
    cfg = _apply_dry_run_overrides(config or ResearchOASISTrainingConfig())
    print("oasis2_trainer: seeding", flush=True)
    set_global_seed(cfg.data.seed, deterministic=cfg.deterministic)
    print("oasis2_trainer: creating run paths", flush=True)
    paths = build_oasis2_run_paths(resolved_settings, cfg.run_name)
    print(f"oasis2_trainer: run_root={paths.run_root}", flush=True)
    print("oasis2_trainer: rebuilding readiness report", flush=True)
    readiness_report = build_oasis2_training_readiness_report(
        resolved_settings,
        seed=cfg.data.seed,
        split_seed=cfg.data.split_seed,
        train_fraction=cfg.data.train_fraction,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
    )
    readiness_json_path, readiness_md_path = save_oasis2_training_readiness_report(
        readiness_report,
        resolved_settings,
    )
    print(
        f"oasis2_trainer: readiness_status={readiness_report.overall_status} "
        f"json={readiness_json_path}",
        flush=True,
    )
    if readiness_report.overall_status != "pass":
        raise ResearchTrainingError(
            "OASIS-2 training is blocked because the supervised readiness gate did not pass. "
            f"See {readiness_md_path} (JSON: {readiness_json_path})."
        )

    print("oasis2_trainer: loading model config", flush=True)
    model_cfg = load_oasis_model_config(cfg.model_config_path)

    print("oasis2_trainer: importing torch symbols", flush=True)
    torch = _load_torch_symbols()["torch"]
    print("oasis2_trainer: resolving device", flush=True)
    device = _resolve_device(cfg.device, torch)
    amp_enabled = bool(cfg.mixed_precision and device.startswith("cuda"))
    print(f"oasis2_trainer: building model on {device}", flush=True)
    model = build_model(model_cfg).to(device)
    print("oasis2_trainer: building optimizer/scheduler/loss", flush=True)
    optimizer = build_optimizer(
        model,
        name=cfg.optimizer.name,
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        momentum=cfg.optimizer.momentum,
    )
    scheduler = build_scheduler(
        optimizer,
        name=cfg.scheduler.name,
        step_size=cfg.scheduler.step_size,
        gamma=cfg.scheduler.gamma,
        patience=cfg.scheduler.patience,
        factor=cfg.scheduler.factor,
    )
    loss_function = build_classification_loss(cfg.loss.name)
    scaler = _build_grad_scaler(torch, amp_enabled=amp_enabled)
    print("oasis2_trainer: building dataloaders", flush=True)
    loader_cfg, dataloaders = _build_loaders(cfg)
    print(
        "oasis2_trainer: dataloaders_ready "
        f"train={len(dataloaders.dataset_bundle.train_records)} "
        f"val={len(dataloaders.dataset_bundle.val_records)} "
        f"test={len(dataloaders.dataset_bundle.test_records)}",
        flush=True,
    )
    split_summary = dict(dataloaders.dataset_bundle.split_artifacts.summary_payload)
    print("oasis2_trainer: saving resolved config", flush=True)
    _save_resolved_oasis2_config(
        cfg=cfg,
        model_cfg=model_cfg,
        paths=paths,
        loader_cfg=loader_cfg,
        split_summary=split_summary,
    )

    start_epoch = 1
    best_monitor_value = _initial_best_value(cfg.early_stopping.mode)
    best_epoch = 0
    if cfg.checkpoint.resume_from is not None:
        start_epoch, best_monitor_value, best_epoch = _load_resume_checkpoint(
            path=cfg.checkpoint.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            torch=torch,
            device=device,
        )

    rows: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    stopped_early = False
    final_val_metrics: dict[str, Any] | None = None
    start_time = perf_counter()
    print(
        f"Starting OASIS-2 training run {cfg.run_name} on device={device} "
        f"for up to {cfg.epochs} epochs. run_root={paths.run_root}"
    )
    print(f"Epoch metrics will be written to {paths.epoch_metrics_csv_path}")
    if cfg.checkpoint.resume_from is not None:
        print(f"Resuming from checkpoint {cfg.checkpoint.resume_from} at epoch {start_epoch}")
    print(f"oasis2_split_report_root={dataloaders.dataset_bundle.split_artifacts.report_root}")
    print(
        f"oasis2_row_counts={json.dumps(dataloaders.dataset_bundle.split_artifacts.summary_payload['row_counts'])}"
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = _run_epoch(
            loader=dataloaders.train_loader,
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            torch=torch,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            max_batches=cfg.data.max_train_batches,
            gradient_accumulation_steps=cfg.data.gradient_accumulation_steps,
        )
        val_metrics = _run_epoch(
            loader=dataloaders.val_loader,
            model=model,
            loss_function=loss_function,
            optimizer=None,
            torch=torch,
            device=device,
            scaler=None,
            amp_enabled=amp_enabled,
            max_batches=cfg.data.max_val_batches,
            gradient_accumulation_steps=1,
        )
        final_val_metrics = val_metrics
        learning_rate = float(optimizer.param_groups[0]["lr"])
        rows.append(_epoch_row(epoch, train_metrics, val_metrics, learning_rate))
        _write_epoch_metrics(rows, paths)
        _step_scheduler(scheduler, cfg.scheduler.name, float(val_metrics["loss"]))

        monitor_value = _resolve_monitor_value(val_metrics, cfg.early_stopping.monitor)
        improved = _is_improvement(
            monitor_value,
            best_monitor_value,
            mode=cfg.early_stopping.mode,
            min_delta=cfg.early_stopping.min_delta,
        )
        if improved:
            best_monitor_value = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_payload = _checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_monitor_value=best_monitor_value,
                best_epoch=best_epoch,
                cfg=cfg,
            )
            if cfg.checkpoint.save_best:
                _save_checkpoint(paths.best_checkpoint_path, checkpoint_payload, torch)
        else:
            epochs_without_improvement += 1

        checkpoint_payload = _checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_monitor_value=best_monitor_value,
            best_epoch=best_epoch,
            cfg=cfg,
        )
        if cfg.checkpoint.save_last:
            _save_checkpoint(paths.last_checkpoint_path, checkpoint_payload, torch)

        improvement_status = "improved" if improved else f"no_improve={epochs_without_improvement}"
        print(
            f"[{cfg.run_name}] epoch {epoch}/{cfg.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"best_epoch={best_epoch} "
            f"best_{cfg.early_stopping.monitor}={best_monitor_value:.4f} "
            f"{improvement_status}"
        )

        if cfg.early_stopping.enabled and epochs_without_improvement >= cfg.early_stopping.patience:
            stopped_early = True
            print(
                f"Early stopping triggered for run {cfg.run_name} at epoch {epoch}. "
                f"Best epoch={best_epoch} best_{cfg.early_stopping.monitor}={best_monitor_value:.4f}"
            )
            break

    if not rows or final_val_metrics is None:
        raise ResearchTrainingError("Training finished without producing epoch metrics.")

    elapsed_seconds = perf_counter() - start_time
    _write_confusion_matrix(final_val_metrics, paths)
    _write_summary_report(
        cfg=cfg,
        paths=paths,
        rows=rows,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor_value,
        stopped_early=stopped_early,
        elapsed_seconds=elapsed_seconds,
        split_summary=split_summary,
    )

    return ResearchTrainingResult(
        run_name=cfg.run_name,
        run_root=paths.run_root,
        best_checkpoint_path=paths.best_checkpoint_path if paths.best_checkpoint_path.exists() else None,
        last_checkpoint_path=paths.last_checkpoint_path if paths.last_checkpoint_path.exists() else None,
        epoch_metrics_csv_path=paths.epoch_metrics_csv_path,
        epoch_metrics_json_path=paths.epoch_metrics_json_path,
        confusion_matrix_path=paths.confusion_matrix_path,
        summary_report_path=paths.summary_report_path,
        resolved_config_path=paths.resolved_config_path,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor_value,
        stopped_early=stopped_early,
        final_metrics=rows[-1],
    )
