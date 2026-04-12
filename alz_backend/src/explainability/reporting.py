"""Reporting helpers for explanation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ExplainabilityArtifact:
    """Represents one saved explainability artifact."""

    method: str
    location: str
    notes: str


def build_stub_explainability_artifact() -> ExplainabilityArtifact:
    """Return a minimal explainability artifact placeholder."""

    return ExplainabilityArtifact(
        method="grad_cam_style_3d",
        location="outputs/visualizations/explanations",
        notes="Explainability is for decision-support review and is not a diagnosis.",
    )


def summarize_heatmap_intensity(heatmap: np.ndarray) -> dict[str, float]:
    """Compute compact summary statistics for a Grad-CAM heatmap."""

    return {
        "mean": float(np.mean(heatmap)),
        "std": float(np.std(heatmap)),
        "max": float(np.max(heatmap)),
        "p95": float(np.percentile(heatmap, 95)),
        "nonzero_fraction": float(np.mean(heatmap > 0)),
    }


def describe_highlighted_regions(region_importance_proxy: dict[str, float]) -> str:
    """Build a human-readable description of the most active coarse regions."""

    if not region_importance_proxy:
        return "No coarse highlighted regions were available."
    ordered = sorted(region_importance_proxy.items(), key=lambda item: item[1], reverse=True)
    top_regions = [name for name, _ in ordered[:2]]
    return "Highest coarse heatmap activity appeared in: " + ", ".join(top_regions) + "."


def interpret_confidence(uncertainty: dict[str, Any], *, confidence_level: str) -> str:
    """Translate uncertainty signals into careful decision-support wording."""

    confidence = float(uncertainty.get("confidence", 0.0))
    entropy = float(uncertainty.get("normalized_entropy", 0.0))
    if confidence_level == "high":
        return (
            f"The model response was relatively concentrated for this scan "
            f"(confidence={confidence:.3f}, normalized_entropy={entropy:.3f})."
        )
    if confidence_level == "medium":
        return (
            f"The model response was moderately concentrated "
            f"(confidence={confidence:.3f}, normalized_entropy={entropy:.3f}); review remains appropriate."
        )
    return (
        f"The model response was uncertain "
        f"(confidence={confidence:.3f}, normalized_entropy={entropy:.3f}); manual review is recommended."
    )


def classify_explanation_quality(
    heatmap_summary: dict[str, float],
    *,
    confidence_level: str,
) -> str:
    """Classify explanation clarity using coarse heatmap and confidence cues."""

    peak = float(heatmap_summary.get("max", 0.0))
    spread = float(heatmap_summary.get("std", 0.0))
    if confidence_level == "high" and peak >= 0.65 and spread >= 0.08:
        return "clear"
    return "uncertain"


def compare_explanation_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare correct vs incorrect explanations when correctness labels exist."""

    grouped: dict[str, list[dict[str, Any]]] = {"correct": [], "incorrect": [], "unknown": []}
    for report in reports:
        correctness = str(report.get("prediction_correctness") or "unknown")
        if correctness not in grouped:
            correctness = "unknown"
        grouped[correctness].append(report)

    def _summaries(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"count": 0, "mean_heatmap_max": 0.0, "quality_counts": {}}
        max_values = [
            float(item.get("heatmap_intensity_summary", {}).get("max", 0.0))
            for item in items
        ]
        quality_counts: dict[str, int] = {}
        for item in items:
            quality = str(item.get("explanation_quality", "unknown"))
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        return {
            "count": len(items),
            "mean_heatmap_max": float(np.mean(max_values)),
            "quality_counts": quality_counts,
        }

    return {
        "correct": _summaries(grouped["correct"]),
        "incorrect": _summaries(grouped["incorrect"]),
        "unknown": _summaries(grouped["unknown"]),
        "notes": [
            "This comparison is descriptive only and does not validate explanation correctness.",
            "Prediction correctness must be provided explicitly to enable correct-vs-incorrect grouping.",
        ],
    }
