"""Tests for the Kaggle Colab training wrapper helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_colab_script_module():
    """Load the Kaggle Colab wrapper script as a module from its file path."""

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_kaggle_colab.py"
    spec = importlib.util.spec_from_file_location("train_kaggle_colab", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_running_in_colab_returns_bool() -> None:
    """Colab detection should return a simple boolean."""

    module = _load_colab_script_module()
    assert isinstance(module._running_in_colab(), bool)


def test_resolve_existing_kaggle_root_prefers_nested_bundle_folder(tmp_path: Path) -> None:
    """The helper should tolerate a Drive root that contains one nested bundle folder."""

    module = _load_colab_script_module()
    drive_root = tmp_path / "kaggle_drive"
    nested = drive_root / "kaggle_alz_upload_bundle"
    (nested / "OriginalDataset").mkdir(parents=True)

    assert module._resolve_existing_kaggle_root(drive_root) == nested.resolve()


def test_maybe_resume_checkpoint_returns_existing_last_checkpoint(tmp_path: Path) -> None:
    """Auto-resume should discover the persisted last checkpoint when present."""

    module = _load_colab_script_module()
    checkpoint_path = tmp_path / "outputs" / "runs" / "kaggle" / "unit_run" / "checkpoints" / "last_model.pt"
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
    run_root = tmp_path / "outputs" / "runs" / "kaggle" / "unit_run"
    checkpoint_path = run_root / "checkpoints" / "last_model.pt"
    signature_path = run_root / "reports" / "colab_requested_signature.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    signature_path.write_text(
        json.dumps(
            {
                "epochs": 12,
                "dry_run": False,
                "dropout_prob": 0.1,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "image_size_2d": [224, 224],
                "image_size_3d": [128, 128, 128],
                "seed": 42,
                "split_random_state": 42,
                "train_fraction": 0.7,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
                "early_stopping_monitor": "val_macro_f1",
                "early_stopping_mode": "max",
                "early_stopping_patience": 3,
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
                "epochs": 20,
                "dry_run": False,
                "dropout_prob": 0.1,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "image_size_2d": (224, 224),
                "image_size_3d": (128, 128, 128),
                "seed": 42,
                "split_random_state": 42,
                "train_fraction": 0.7,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
                "early_stopping_monitor": "val_macro_f1",
                "early_stopping_mode": "max",
                "early_stopping_patience": 3,
            },
        )
    except ValueError as exc:
        assert "different training signature" in str(exc)
    else:
        raise AssertionError("Expected mismatched resume signature to raise ValueError")


def test_maybe_reuse_completed_run_uses_existing_final_metrics(tmp_path: Path) -> None:
    """A completed persisted run should be reusable without retraining."""

    module = _load_colab_script_module()
    run_root = tmp_path / "outputs" / "runs" / "kaggle" / "unit_run"
    best_checkpoint = run_root / "checkpoints" / "best_model.pt"
    last_checkpoint = run_root / "checkpoints" / "last_model.pt"
    epoch_metrics = run_root / "metrics" / "epoch_metrics.csv"
    final_metrics = run_root / "metrics" / "final_metrics.json"
    resolved_config = run_root / "configs" / "resolved_config.json"
    signature_path = run_root / "reports" / "colab_requested_signature.json"

    for path in (best_checkpoint, last_checkpoint, epoch_metrics, final_metrics, resolved_config, signature_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint.write_bytes(b"best")
    last_checkpoint.write_bytes(b"last")
    epoch_metrics.write_text("epoch,train_loss,val_loss\n1,0.8,0.7\n", encoding="utf-8")
    final_metrics.write_text(
        json.dumps(
            {
                "run_name": "unit_run",
                "validation": {"accuracy": 0.7, "macro_f1": 0.68, "macro_ovr_auroc": 0.8},
                "test": {"accuracy": 0.72, "macro_f1": 0.7, "macro_ovr_auroc": 0.81},
            }
        ),
        encoding="utf-8",
    )
    resolved_config.write_text(json.dumps({"training": {}}), encoding="utf-8")
    signature_path.write_text(
        json.dumps(
            {
                "epochs": 12,
                "dry_run": False,
                "dropout_prob": 0.1,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "image_size_2d": [224, 224],
                "image_size_3d": [128, 128, 128],
                "seed": 42,
                "split_random_state": 42,
                "train_fraction": 0.7,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
                "early_stopping_monitor": "val_macro_f1",
                "early_stopping_mode": "max",
                "early_stopping_patience": 3,
            }
        ),
        encoding="utf-8",
    )

    reused = module._maybe_reuse_completed_run(
        outputs_root=tmp_path / "outputs",
        run_name="unit_run",
        requested_signature={
            "epochs": 12,
            "dry_run": False,
            "dropout_prob": 0.1,
            "batch_size": 8,
            "gradient_accumulation_steps": 1,
            "image_size_2d": (224, 224),
            "image_size_3d": (128, 128, 128),
            "seed": 42,
            "split_random_state": 42,
            "train_fraction": 0.7,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "early_stopping_monitor": "val_macro_f1",
            "early_stopping_mode": "max",
            "early_stopping_patience": 3,
        },
    )

    assert reused is not None
    assert reused["best_checkpoint"] == best_checkpoint
    assert reused["final_metrics"]["validation"]["macro_ovr_auroc"] == 0.8


def test_maybe_reuse_completed_run_rejects_mismatched_signature(tmp_path: Path) -> None:
    """Completed runs should not be reused when the requested config changed."""

    module = _load_colab_script_module()
    run_root = tmp_path / "outputs" / "runs" / "kaggle" / "unit_run"
    best_checkpoint = run_root / "checkpoints" / "best_model.pt"
    epoch_metrics = run_root / "metrics" / "epoch_metrics.csv"
    final_metrics = run_root / "metrics" / "final_metrics.json"
    resolved_config = run_root / "configs" / "resolved_config.json"
    signature_path = run_root / "reports" / "colab_requested_signature.json"

    for path in (best_checkpoint, epoch_metrics, final_metrics, resolved_config, signature_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint.write_bytes(b"best")
    epoch_metrics.write_text("epoch,train_loss,val_loss\n1,0.8,0.7\n", encoding="utf-8")
    final_metrics.write_text(json.dumps({"validation": {}, "test": {}}), encoding="utf-8")
    resolved_config.write_text(json.dumps({"training": {}}), encoding="utf-8")
    signature_path.write_text(
        json.dumps(
            {
                "epochs": 12,
                "dry_run": False,
                "dropout_prob": 0.1,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "image_size_2d": [224, 224],
                "image_size_3d": [128, 128, 128],
                "seed": 42,
                "split_random_state": 42,
                "train_fraction": 0.7,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
                "early_stopping_monitor": "val_macro_f1",
                "early_stopping_mode": "max",
                "early_stopping_patience": 3,
            }
        ),
        encoding="utf-8",
    )

    try:
        module._maybe_reuse_completed_run(
            outputs_root=tmp_path / "outputs",
            run_name="unit_run",
            requested_signature={
                "epochs": 24,
                "dry_run": False,
                "dropout_prob": 0.1,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "image_size_2d": (224, 224),
                "image_size_3d": (128, 128, 128),
                "seed": 42,
                "split_random_state": 42,
                "train_fraction": 0.7,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
                "early_stopping_monitor": "val_macro_f1",
                "early_stopping_mode": "max",
                "early_stopping_patience": 3,
            },
        )
    except ValueError as exc:
        assert "different training signature" in str(exc)
    else:
        raise AssertionError("Expected mismatched completed-run signature to raise ValueError")
