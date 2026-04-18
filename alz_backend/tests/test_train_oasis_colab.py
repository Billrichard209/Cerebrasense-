"""Tests for the Colab training wrapper helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_colab_script_module():
    """Load the Colab wrapper script as a module from its file path."""

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_oasis_colab.py"
    spec = importlib.util.spec_from_file_location("train_oasis_colab", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_running_in_colab_returns_bool() -> None:
    """Colab detection should return a simple boolean."""

    module = _load_colab_script_module()
    assert isinstance(module._running_in_colab(), bool)


def test_colab_config_exists() -> None:
    """The Colab GPU config should be present in the repository."""

    config_path = Path(__file__).resolve().parents[1] / "configs" / "oasis_train_colab_gpu.yaml"
    assert config_path.exists()


def test_resolve_existing_oasis_root_prefers_nested_folder(tmp_path: Path) -> None:
    """The helper should tolerate a Drive root that contains a nested `OASIS/` folder."""

    module = _load_colab_script_module()
    drive_root = tmp_path / "OASIS-1"
    nested = drive_root / "OASIS"
    nested.mkdir(parents=True)

    assert module._resolve_existing_oasis_root(drive_root) == nested.resolve()


def test_maybe_resume_checkpoint_returns_existing_last_checkpoint(tmp_path: Path) -> None:
    """Auto-resume should discover the persisted last checkpoint when present."""

    module = _load_colab_script_module()
    checkpoint_path = tmp_path / "outputs" / "runs" / "oasis" / "unit_run" / "checkpoints" / "last_model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")

    resolved = module._maybe_resume_checkpoint(
        outputs_root=tmp_path / "outputs",
        run_name="unit_run",
        enabled=True,
    )

    assert resolved == checkpoint_path


def test_maybe_resume_checkpoint_rejects_mismatched_existing_signature(tmp_path: Path) -> None:
    """Resume should fail loudly when the persisted run name belongs to a different config."""

    module = _load_colab_script_module()
    run_root = tmp_path / "outputs" / "runs" / "oasis" / "unit_run"
    checkpoint_path = run_root / "checkpoints" / "last_model.pt"
    resolved_config_path = run_root / "configs" / "resolved_config.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    resolved_config_path.write_text(
        json.dumps(
            {
                "training": {
                    "epochs": 20,
                    "data": {
                        "batch_size": 2,
                        "gradient_accumulation_steps": 1,
                        "image_size": [64, 64, 64],
                        "seed": 42,
                        "split_seed": 42,
                        "weighted_sampling": False,
                    },
                    "early_stopping": {
                        "monitor": "val_loss",
                        "mode": "min",
                        "patience": 5,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        module._maybe_resume_checkpoint(
            outputs_root=tmp_path / "outputs",
            run_name="unit_run",
            enabled=True,
            requested_signature={
                "epochs": 28,
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "image_size": (64, 64, 64),
                "seed": 42,
                "split_seed": 42,
                "weighted_sampling": False,
                "early_stopping_monitor": "val_auroc",
                "early_stopping_mode": "max",
                "early_stopping_patience": 6,
                "evaluate_splits": ("val", "test"),
            },
        )
    except ValueError as exc:
        assert "different training signature" in str(exc)
    else:
        raise AssertionError("Expected mismatched resume signature to raise ValueError")


def test_maybe_reuse_completed_run_uses_existing_summary_and_eval_artifacts(tmp_path: Path) -> None:
    """A completed persisted run should be reusable without retraining."""

    module = _load_colab_script_module()
    run_root = tmp_path / "outputs" / "runs" / "oasis" / "unit_run"
    best_checkpoint = run_root / "checkpoints" / "best_model.pt"
    last_checkpoint = run_root / "checkpoints" / "last_model.pt"
    val_metrics = run_root / "evaluation" / "post_train_val_best_model" / "metrics.json"
    val_predictions = run_root / "evaluation" / "post_train_val_best_model" / "predictions.csv"
    report_path = run_root / "reports" / "experiment_summary.json"

    for path in (best_checkpoint, last_checkpoint, val_metrics, val_predictions, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint.write_bytes(b"best")
    last_checkpoint.write_bytes(b"last")
    val_metrics.write_text("{}", encoding="utf-8")
    val_predictions.write_text("true_label,probability_class_1\n0,0.2\n", encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {
                "training": {
                    "best_checkpoint": str(best_checkpoint),
                    "last_checkpoint": str(last_checkpoint),
                },
                "config": {
                    "training": {
                        "epochs": 28,
                        "data": {
                            "batch_size": 4,
                            "gradient_accumulation_steps": 1,
                            "image_size": [64, 64, 64],
                            "seed": 42,
                            "split_seed": 42,
                            "weighted_sampling": False,
                        },
                        "early_stopping": {
                            "monitor": "val_auroc",
                            "mode": "max",
                            "patience": 6,
                        },
                    },
                    "evaluate_splits": ["val"],
                },
                "evaluations": {
                    "val": {
                        "metrics": {"accuracy": 0.75, "auroc": 0.8},
                        "metrics_json_path": str(val_metrics),
                        "predictions_csv_path": str(val_predictions),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    reused = module._maybe_reuse_completed_run(
        outputs_root=tmp_path / "outputs",
        run_name="unit_run",
        evaluate_splits=("val",),
        requested_signature={
            "epochs": 28,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "image_size": (64, 64, 64),
            "seed": 42,
            "split_seed": 42,
            "weighted_sampling": False,
            "early_stopping_monitor": "val_auroc",
            "early_stopping_mode": "max",
            "early_stopping_patience": 6,
            "evaluate_splits": ("val",),
        },
    )

    assert reused is not None
    assert reused["best_checkpoint"] == best_checkpoint
    assert reused["evaluations"]["val"]["metrics_json_path"] == val_metrics


def test_maybe_reuse_completed_run_rejects_mismatched_signature(tmp_path: Path) -> None:
    """Completed runs should not be reused when the requested config changed."""

    module = _load_colab_script_module()
    run_root = tmp_path / "outputs" / "runs" / "oasis" / "unit_run"
    val_metrics = run_root / "evaluation" / "post_train_val_best_model" / "metrics.json"
    val_predictions = run_root / "evaluation" / "post_train_val_best_model" / "predictions.csv"
    report_path = run_root / "reports" / "experiment_summary.json"
    for path in (val_metrics, val_predictions, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    val_metrics.write_text("{}", encoding="utf-8")
    val_predictions.write_text("true_label,probability_class_1\n0,0.2\n", encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {
                "config": {
                    "training": {
                        "epochs": 20,
                        "data": {
                            "batch_size": 4,
                            "gradient_accumulation_steps": 1,
                            "image_size": [64, 64, 64],
                            "seed": 42,
                            "split_seed": 42,
                            "weighted_sampling": False,
                        },
                        "early_stopping": {
                            "monitor": "val_loss",
                            "mode": "min",
                            "patience": 5,
                        },
                    },
                    "evaluate_splits": ["val"],
                },
                "evaluations": {
                    "val": {
                        "metrics": {"accuracy": 0.75, "auroc": 0.8},
                        "metrics_json_path": str(val_metrics),
                        "predictions_csv_path": str(val_predictions),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    reused = module._maybe_reuse_completed_run(
        outputs_root=tmp_path / "outputs",
        run_name="unit_run",
        evaluate_splits=("val",),
        requested_signature={
            "epochs": 28,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "image_size": (64, 64, 64),
            "seed": 42,
            "split_seed": 42,
            "weighted_sampling": False,
            "early_stopping_monitor": "val_auroc",
            "early_stopping_mode": "max",
            "early_stopping_patience": 6,
            "evaluate_splits": ("val",),
        },
    )

    assert reused is None
