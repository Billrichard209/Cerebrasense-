"""Classification metrics and uncertainty helpers for model evaluation."""

from __future__ import annotations

from math import log
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
    """Compute simple accuracy for quick smoke tests and placeholder evaluations."""

    if not y_true:
        return 0.0
    correct = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == predicted)
    return correct / len(y_true)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return a safe ratio for metric calculations."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_confusion_counts(
    y_true: list[int],
    y_pred: list[int],
    *,
    positive_label: int = 1,
    negative_label: int = 0,
) -> dict[str, int]:
    """Compute binary confusion counts using explicit positive/negative labels."""

    if len(y_true) != len(y_pred):
        raise ValueError(f"Expected y_true and y_pred to have the same length, got {len(y_true)} and {len(y_pred)}.")

    true_positive = sum(1 for truth, pred in zip(y_true, y_pred) if truth == positive_label and pred == positive_label)
    true_negative = sum(1 for truth, pred in zip(y_true, y_pred) if truth == negative_label and pred == negative_label)
    false_positive = sum(1 for truth, pred in zip(y_true, y_pred) if truth == negative_label and pred == positive_label)
    false_negative = sum(1 for truth, pred in zip(y_true, y_pred) if truth == positive_label and pred == negative_label)
    return {
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }


def compute_binary_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    *,
    y_score: list[float] | None = None,
    positive_label: int = 1,
    negative_label: int = 0,
) -> dict[str, float | int | dict[str, int]]:
    """Compute baseline binary classification metrics for OASIS-style outputs."""

    counts = compute_confusion_counts(
        y_true,
        y_pred,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    tp = counts["true_positive"]
    tn = counts["true_negative"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    total = tp + tn + fp + fn
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "sample_count": int(total),
        "accuracy": _safe_divide(tp + tn, total),
        "auroc": compute_binary_auroc(y_true, y_score) if y_score is not None else 0.0,
        "precision": precision,
        "recall_sensitivity": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_counts": counts,
    }


def compute_binary_auroc(y_true: list[int], y_score: list[float] | None) -> float:
    """Compute binary AUROC, returning 0.0 when undefined for tiny splits."""

    if not y_true or y_score is None:
        return 0.0
    if len(set(y_true)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.0


def threshold_binary_scores(y_score: list[float], *, threshold: float = 0.5) -> list[int]:
    """Convert positive-class probabilities into binary predictions."""

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"Classification threshold must be between 0 and 1, got {threshold}.")
    return [1 if float(score) >= threshold else 0 for score in y_score]


def compute_binary_roc_curve(
    y_true: list[int],
    y_score: list[float],
) -> dict[str, Any]:
    """Compute ROC curve points and AUROC for binary classification.

    When a split has only one class, ROC is mathematically undefined. In that
    case this returns a diagonal placeholder curve with ``is_defined=False`` so
    reports can still be generated without pretending the AUROC is meaningful.
    """

    if len(y_true) != len(y_score):
        raise ValueError(f"Expected y_true and y_score to have the same length, got {len(y_true)} and {len(y_score)}.")
    if not y_true:
        return {
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "auroc": 0.0,
            "is_defined": False,
            "warning": "ROC curve is undefined because no samples were provided.",
        }
    if len(set(y_true)) < 2:
        return {
            "fpr": [0.0, 1.0],
            "tpr": [0.0, 1.0],
            "thresholds": [float("inf"), float("-inf")],
            "auroc": 0.0,
            "is_defined": False,
            "warning": "ROC curve is undefined because the evaluated split contains only one class.",
        }

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {
        "fpr": [float(value) for value in fpr.tolist()],
        "tpr": [float(value) for value in tpr.tolist()],
        "thresholds": [float(value) for value in thresholds.tolist()],
        "auroc": compute_binary_auroc(y_true, y_score),
        "is_defined": True,
        "warning": None,
    }


def build_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    *,
    negative_label: int = 0,
    positive_label: int = 1,
) -> list[list[int]]:
    """Return a 2x2 confusion matrix as [[TN, FP], [FN, TP]]."""

    counts = compute_confusion_counts(
        y_true,
        y_pred,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    return [
        [counts["true_negative"], counts["false_positive"]],
        [counts["false_negative"], counts["true_positive"]],
    ]


def normalize_probabilities(probabilities: Any) -> np.ndarray:
    """Convert probability-like input to a validated 2D NumPy array."""

    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim == 1:
        probability_array = probability_array.reshape(1, -1)
    if probability_array.ndim != 2:
        raise ValueError(f"Expected probabilities shaped (N, C), got {probability_array.shape}.")
    if probability_array.shape[1] < 2:
        raise ValueError("At least two classes are required for classification uncertainty scoring.")
    row_sums = probability_array.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("Probability rows must have positive sums.")
    return probability_array / row_sums[:, None]


def compute_uncertainty_from_probabilities(probabilities: Any) -> list[dict[str, float]]:
    """Compute confidence, entropy, normalized entropy, and margin for each sample."""

    probability_array = normalize_probabilities(probabilities)
    clipped = np.clip(probability_array, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    class_count = probability_array.shape[1]
    normalized_entropy = entropy / log(class_count)
    sorted_probs = np.sort(probability_array, axis=1)
    top_probability = sorted_probs[:, -1]
    second_probability = sorted_probs[:, -2]
    margin = top_probability - second_probability
    uncertainty_score = 1.0 - top_probability
    return [
        {
            "confidence": float(top_probability[index]),
            "entropy": float(entropy[index]),
            "normalized_entropy": float(normalized_entropy[index]),
            "probability_margin": float(margin[index]),
            "uncertainty_score": float(uncertainty_score[index]),
        }
        for index in range(probability_array.shape[0])
    ]
