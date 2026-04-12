"""Error-analysis helpers for OASIS classification experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ErrorAnalysisConfig:
    """Configuration for post-hoc misclassification analysis."""

    predictions_csv_path: Path
    output_name: str = "oasis_error_analysis"
    max_examples_per_bucket: int | None = None
    save_slices: bool = True


@dataclass(slots=True)
class ErrorAnalysisPaths:
    """Resolved output paths for one error-analysis run."""

    output_root: Path
    summary_json_path: Path
    misclassifications_csv_path: Path


@dataclass(slots=True)
class ErrorAnalysisResult:
    """Saved error-analysis summary plus output paths."""

    config: ErrorAnalysisConfig
    paths: ErrorAnalysisPaths
    summary: dict[str, Any]


def _resolve_output_paths(
    cfg: ErrorAnalysisConfig,
    *,
    settings: AppSettings | None = None,
) -> ErrorAnalysisPaths:
    """Build error-analysis output paths."""

    resolved_settings = settings or get_app_settings()
    safe_name = cfg.output_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "errors" / safe_name)
    return ErrorAnalysisPaths(
        output_root=output_root,
        summary_json_path=output_root / "error_summary.json",
        misclassifications_csv_path=output_root / "misclassified_samples.csv",
    )


def _load_predictions(path: Path) -> pd.DataFrame:
    """Load predictions CSV and validate minimum fields."""

    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")
    frame = pd.read_csv(path)
    required_groups = {
        "true_label": {"true_label"},
        "predicted_label": {"predicted_label"},
        "probability": {"probability", "probability_class_1", "calibrated_probability_score"},
    }
    for label, candidates in required_groups.items():
        if not any(candidate in frame.columns for candidate in candidates):
            raise ValueError(f"Predictions CSV is missing a required {label} column. Expected one of {sorted(candidates)}.")
    return frame


def _pick_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first available column name from the supplied candidates."""

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _normalize_probability(frame: pd.DataFrame) -> pd.Series:
    """Resolve the positive-class probability column."""

    column = _pick_column(frame, "calibrated_probability_score", "probability", "probability_class_1")
    if column is None:
        raise ValueError("Could not resolve a probability column for error analysis.")
    return frame[column].astype(float)


def _normalize_confidence(frame: pd.DataFrame) -> pd.Series:
    """Resolve a confidence-like column, falling back to probability."""

    column = _pick_column(frame, "confidence", "probability", "probability_class_1", "calibrated_probability_score")
    if column is None:
        raise ValueError("Could not resolve a confidence column for error analysis.")
    return frame[column].astype(float)


def _normalize_source_path(frame: pd.DataFrame) -> pd.Series:
    """Resolve the source-image path column."""

    column = _pick_column(frame, "source_path", "meta_image_path", "image_path")
    if column is None:
        return pd.Series([""] * len(frame))
    return frame[column].fillna("").astype(str)


def _normalize_sample_id(frame: pd.DataFrame) -> pd.Series:
    """Resolve a stable sample identifier."""

    column = _pick_column(frame, "sample_id", "subject_id", "meta_subject_id", "session_id", "meta_session_id")
    if column is None:
        return pd.Series([f"sample_{index:05d}" for index in range(len(frame))])
    return frame[column].fillna("").astype(str).replace("", np.nan).fillna(
        pd.Series([f"sample_{index:05d}" for index in range(len(frame))])
    )


def _confidence_distribution(values: list[float]) -> dict[str, int]:
    """Bucket confidence scores for compact reporting."""

    bins = {
        "0.00-0.50": 0,
        "0.50-0.70": 0,
        "0.70-0.85": 0,
        "0.85-1.00": 0,
    }
    for value in values:
        if value < 0.50:
            bins["0.00-0.50"] += 1
        elif value < 0.70:
            bins["0.50-0.70"] += 1
        elif value < 0.85:
            bins["0.70-0.85"] += 1
        else:
            bins["0.85-1.00"] += 1
    return bins


def _class_imbalance_insights(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize the true-label balance of the evaluated split."""

    true_counts = frame["true_label"].astype(int).value_counts().sort_index().to_dict()
    negative_count = int(true_counts.get(0, 0))
    positive_count = int(true_counts.get(1, 0))
    majority = max(negative_count, positive_count, 1)
    minority = max(min(negative_count, positive_count), 1)
    ratio = float(majority / minority)
    if ratio < 1.2:
        note = "Classes are close to balanced in this evaluated split."
    elif ratio < 2.0:
        note = "Classes are moderately imbalanced; inspect error rates by class carefully."
    else:
        note = "Classes are strongly imbalanced; headline accuracy may hide poor minority-class behavior."
    return {
        "true_label_counts": {"0": negative_count, "1": positive_count},
        "imbalance_ratio": ratio,
        "insight": note,
    }


def _load_volume(path: Path) -> np.ndarray:
    """Load a 3D volume from an MRI path."""

    import nibabel as nib

    image = nib.load(str(path))
    array = np.asarray(image.get_fdata())
    if array.ndim == 4 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {array.shape} for {path}.")
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value <= min_value:
        return np.zeros_like(array, dtype=float)
    return (array - min_value) / (max_value - min_value)


def _save_slice_triplet(volume: np.ndarray, output_dir: Path) -> list[str]:
    """Save axial/coronal/sagittal middle slices."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    slice_specs = {
        "axial_mid.png": volume[volume.shape[0] // 2, :, :],
        "coronal_mid.png": volume[:, volume.shape[1] // 2, :],
        "sagittal_mid.png": volume[:, :, volume.shape[2] // 2],
    }
    saved_paths: list[str] = []
    for file_name, slice_array in slice_specs.items():
        output_path = output_dir / file_name
        fig, axis = plt.subplots(figsize=(4.5, 4.5))
        axis.imshow(np.rot90(slice_array), cmap="gray")
        axis.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        saved_paths.append(str(output_path))
    return saved_paths


def _save_error_artifact(row: pd.Series, *, category: str, output_root: Path, save_slices: bool) -> tuple[list[str], list[str]]:
    """Save one misclassification artifact bundle."""

    sample_dir = ensure_directory(output_root / category / str(row["sample_id"]))
    warnings: list[str] = []
    slice_paths: list[str] = []
    source_path = Path(str(row["source_path"])) if str(row["source_path"]).strip() else None
    if save_slices and source_path is not None and source_path.exists():
        try:
            slice_paths = _save_slice_triplet(_load_volume(source_path), sample_dir)
        except Exception as error:  # pragma: no cover - defensive filesystem/data path
            warnings.append(f"Could not save slices for {source_path}: {error}")
            LOGGER.warning("Could not save slices for %s: %s", source_path, error)
    elif save_slices and source_path is not None:
        warnings.append(f"Source image not found: {source_path}")
    metadata = {
        "sample_id": row["sample_id"],
        "category": category,
        "subject_id": row.get("subject_id") or row.get("meta_subject_id") or "",
        "session_id": row.get("session_id") or row.get("meta_session_id") or "",
        "source_path": str(source_path) if source_path is not None else "",
        "true_label": int(row["true_label"]),
        "predicted_label": int(row["predicted_label"]),
        "probability": float(row["probability"]),
        "confidence": float(row["confidence"]),
        "slice_paths": slice_paths,
        "warnings": warnings,
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return slice_paths, warnings


def analyze_prediction_errors(
    cfg: ErrorAnalysisConfig,
    *,
    settings: AppSettings | None = None,
) -> ErrorAnalysisResult:
    """Analyze false positives and false negatives from a predictions CSV."""

    frame = _load_predictions(cfg.predictions_csv_path).copy()
    frame["true_label"] = frame["true_label"].astype(int)
    frame["predicted_label"] = frame["predicted_label"].astype(int)
    frame["probability"] = _normalize_probability(frame)
    frame["confidence"] = _normalize_confidence(frame)
    frame["source_path"] = _normalize_source_path(frame)
    frame["sample_id"] = _normalize_sample_id(frame)
    frame["error_category"] = np.where(
        (frame["true_label"] == 0) & (frame["predicted_label"] == 1),
        "false_positive",
        np.where(
            (frame["true_label"] == 1) & (frame["predicted_label"] == 0),
            "false_negative",
            "",
        ),
    )
    errors = frame[frame["error_category"] != ""].copy()
    if cfg.max_examples_per_bucket is not None:
        errors = (
            errors.groupby("error_category", group_keys=False)
            .head(int(cfg.max_examples_per_bucket))
            .reset_index(drop=True)
        )
    paths = _resolve_output_paths(cfg, settings=settings)

    warnings: list[str] = []
    saved_rows: list[dict[str, Any]] = []
    for _, row in errors.iterrows():
        slice_paths, row_warnings = _save_error_artifact(
            row,
            category=str(row["error_category"]),
            output_root=paths.output_root,
            save_slices=cfg.save_slices,
        )
        warnings.extend(row_warnings)
        saved_row = row.to_dict()
        saved_row["slice_paths"] = slice_paths
        saved_row["warnings"] = row_warnings
        saved_rows.append(saved_row)

    misclassified_frame = pd.DataFrame(saved_rows)
    misclassified_frame.to_csv(paths.misclassifications_csv_path, index=False)

    error_confidences = errors["confidence"].astype(float).tolist()
    summary = {
        "analysis_name": cfg.output_name,
        "predictions_csv_path": str(cfg.predictions_csv_path),
        "sample_count": int(len(frame)),
        "misclassification_count": int(len(errors)),
        "total_false_positives": int((errors["error_category"] == "false_positive").sum()),
        "total_false_negatives": int((errors["error_category"] == "false_negative").sum()),
        "average_confidence_of_errors": float(np.mean(error_confidences)) if error_confidences else 0.0,
        "error_confidence_distribution": _confidence_distribution(error_confidences),
        "class_imbalance_insights": _class_imbalance_insights(frame),
        "output_root": str(paths.output_root),
        "misclassifications_csv_path": str(paths.misclassifications_csv_path),
        "warnings": warnings,
        "notes": [
            "Misclassification analysis is a research debugging aid, not a clinical conclusion.",
            "Saved slices are representative views for review, not anatomical segmentation outputs.",
        ],
    }
    paths.summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ErrorAnalysisResult(
        config=cfg,
        paths=paths,
        summary=summary,
    )
