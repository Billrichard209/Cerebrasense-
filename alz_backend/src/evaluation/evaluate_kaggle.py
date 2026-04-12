"""Evaluation entry point placeholder for Kaggle experiments."""

from __future__ import annotations

from .metrics import accuracy_score


def evaluate_kaggle_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Return starter evaluation metrics for Kaggle predictions."""

    return {"accuracy": accuracy_score(y_true, y_pred)}
