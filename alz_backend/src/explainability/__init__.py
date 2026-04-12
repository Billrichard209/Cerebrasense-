"""Explainability utilities for human-readable decision-support artifacts."""

from .gradcam import ExplainScanConfig, ExplainabilityError, ExplanationResult, explain_scan
from .reporting import compare_explanation_reports

__all__ = [
    "ExplainScanConfig",
    "ExplainabilityError",
    "ExplanationResult",
    "compare_explanation_reports",
    "explain_scan",
]
