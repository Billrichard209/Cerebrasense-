"""Confidence calibration helpers for research-grade model outputs.

This module intentionally keeps calibration lightweight and explicit:
- temperature scaling is configurable but not implicitly learned here
- confidence bands are simple research heuristics, not clinical thresholds
- review flags are conservative and designed for decision-support workflows
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .metrics import compute_uncertainty_from_probabilities, normalize_probabilities


@dataclass(slots=True, frozen=True)
class ConfidenceBandConfig:
    """Simple confidence-band thresholds for backend decision support.

    ``temperature`` supports post-hoc temperature scaling when a calibrated
    value is known. A value of ``1.0`` means no scaling is applied.
    """

    temperature: float = 1.0
    high_confidence_min: float = 0.85
    medium_confidence_min: float = 0.65
    high_entropy_max: float = 0.35
    medium_entropy_max: float = 0.90

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-safe representation."""

        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(slots=True, frozen=True)
class CalibratedConfidence:
    """Calibrated confidence summary for one sample."""

    calibrated_probabilities: list[float]
    calibrated_probability_score: float
    confidence_score: float
    confidence_level: str
    review_flag: bool
    temperature: float
    normalized_entropy: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        payload = asdict(self)
        payload["calibrated_probabilities"] = [float(value) for value in self.calibrated_probabilities]
        return payload


def temperature_scale_probabilities(probabilities: Any, *, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to probability rows.

    This function rescales probabilities through log-space and softmax. It is a
    practical scaffold for post-hoc calibration without claiming the supplied
    temperature was learned or clinically validated.
    """

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}.")
    probability_array = normalize_probabilities(probabilities)
    if np.isclose(temperature, 1.0):
        return probability_array
    logits = np.log(np.clip(probability_array, 1e-12, 1.0))
    scaled_logits = logits / float(temperature)
    shifted = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def classify_confidence_level(
    *,
    confidence_score: float,
    normalized_entropy: float,
    config: ConfidenceBandConfig | None = None,
) -> str:
    """Classify one prediction into high, medium, or low confidence."""

    resolved_config = config or ConfidenceBandConfig()
    if (
        confidence_score >= resolved_config.high_confidence_min
        and normalized_entropy <= resolved_config.high_entropy_max
    ):
        return "high"
    if (
        confidence_score >= resolved_config.medium_confidence_min
        and normalized_entropy <= resolved_config.medium_entropy_max
    ):
        return "medium"
    return "low"


def summarize_calibrated_confidence(
    probabilities: Any,
    *,
    positive_class_index: int = 1,
    config: ConfidenceBandConfig | None = None,
) -> list[CalibratedConfidence]:
    """Build calibrated confidence summaries for probability rows."""

    resolved_config = config or ConfidenceBandConfig()
    calibrated = temperature_scale_probabilities(probabilities, temperature=resolved_config.temperature)
    uncertainty_rows = compute_uncertainty_from_probabilities(calibrated)
    summaries: list[CalibratedConfidence] = []
    for index, row in enumerate(calibrated.tolist()):
        probability_score = float(row[positive_class_index]) if positive_class_index < len(row) else float(max(row))
        confidence_score = float(max(row))
        normalized_entropy = float(uncertainty_rows[index]["normalized_entropy"])
        confidence_level = classify_confidence_level(
            confidence_score=confidence_score,
            normalized_entropy=normalized_entropy,
            config=resolved_config,
        )
        summaries.append(
            CalibratedConfidence(
                calibrated_probabilities=[float(value) for value in row],
                calibrated_probability_score=probability_score,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                review_flag=confidence_level == "low",
                temperature=float(resolved_config.temperature),
                normalized_entropy=normalized_entropy,
            )
        )
    return summaries
