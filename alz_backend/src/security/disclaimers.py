"""Standard decision-support disclaimers for model/report outputs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

STANDARD_DECISION_SUPPORT_DISCLAIMER = (
    "This output is for research and clinical decision support only. "
    "It is not a diagnosis, does not replace qualified clinical judgment, "
    "and must be reviewed with the source imaging and clinical context."
)

EXPLANATION_LIMITATION = (
    "Explainability heatmaps are approximate model-attribution artifacts, "
    "not anatomical segmentations or proof of pathology."
)


def build_ai_summary(*, label_name: str, probability_score: float, confidence_score: float) -> str:
    """Build standard model-output wording that avoids diagnosis claims."""

    return (
        f"The model output is {label_name!r} with positive-class probability "
        f"{probability_score:.3f} and confidence {confidence_score:.3f}. "
        f"{STANDARD_DECISION_SUPPORT_DISCLAIMER}"
    )


def add_decision_support_disclaimer(
    payload: dict[str, Any],
    *,
    field_name: str = "clinical_disclaimer",
    mutate: bool = False,
) -> dict[str, Any]:
    """Attach the standard disclaimer to a payload."""

    resolved_payload = payload if mutate else deepcopy(payload)
    resolved_payload[field_name] = STANDARD_DECISION_SUPPORT_DISCLAIMER
    resolved_payload["decision_support_only"] = True
    notes = resolved_payload.setdefault("notes", [])
    if isinstance(notes, list) and STANDARD_DECISION_SUPPORT_DISCLAIMER not in notes:
        notes.append(STANDARD_DECISION_SUPPORT_DISCLAIMER)
    return resolved_payload

