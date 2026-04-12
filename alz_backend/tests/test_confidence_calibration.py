"""Tests for confidence calibration helpers."""

from __future__ import annotations

import pytest

from src.evaluation.calibration import (
    ConfidenceBandConfig,
    classify_confidence_level,
    summarize_calibrated_confidence,
    temperature_scale_probabilities,
)


def test_temperature_scaling_preserves_shape_and_normalization() -> None:
    """Temperature scaling should return normalized probability rows."""

    scaled = temperature_scale_probabilities([[0.2, 0.8], [0.5, 0.5]], temperature=1.5)

    assert scaled.shape == (2, 2)
    assert scaled[0].sum() == pytest.approx(1.0)
    assert scaled[1].sum() == pytest.approx(1.0)


def test_confidence_levels_are_assigned_conservatively() -> None:
    """Confidence bands should distinguish high, medium, and low cases."""

    config = ConfidenceBandConfig(
        temperature=1.0,
        high_confidence_min=0.85,
        medium_confidence_min=0.65,
        high_entropy_max=0.35,
        medium_entropy_max=0.75,
    )

    assert classify_confidence_level(confidence_score=0.92, normalized_entropy=0.20, config=config) == "high"
    assert classify_confidence_level(confidence_score=0.70, normalized_entropy=0.40, config=config) == "medium"
    assert classify_confidence_level(confidence_score=0.58, normalized_entropy=0.90, config=config) == "low"


def test_summarize_calibrated_confidence_builds_review_flags() -> None:
    """Low-confidence predictions should be marked for review."""

    rows = summarize_calibrated_confidence([[0.95, 0.05], [0.5, 0.5]])

    assert rows[0].confidence_level == "high"
    assert rows[0].review_flag is False
    assert rows[1].confidence_level == "low"
    assert rows[1].review_flag is True
