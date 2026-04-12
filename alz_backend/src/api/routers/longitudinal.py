"""Longitudinal tracking API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.auth import AuthContext, require_api_key
from src.api.schemas import LongitudinalReportRequest, LongitudinalReportResponse, OASISStructuralLongitudinalResponse
from src.api.services import build_oasis_longitudinal_structural_payload, build_saved_longitudinal_report_payload

router = APIRouter(tags=["longitudinal"])


@router.get(
    "/longitudinal/oasis/{subject_id}/volumetrics",
    response_model=OASISStructuralLongitudinalResponse,
)
def oasis_longitudinal_volumetrics(
    subject_id: str,
    split: str | None = None,
    manifest_path: str | None = None,
    max_timepoints: int | None = None,
    _auth: AuthContext = Depends(require_api_key),
) -> OASISStructuralLongitudinalResponse:
    """Return subject-level structural proxy change metrics across OASIS visits."""

    return OASISStructuralLongitudinalResponse(
        **build_oasis_longitudinal_structural_payload(
            subject_id=subject_id,
            split=split,
            manifest_path=manifest_path,
            max_timepoints=max_timepoints,
        )
    )


@router.post("/longitudinal/report", response_model=LongitudinalReportResponse)
def build_longitudinal_report_route(
    request: LongitudinalReportRequest,
    _auth: AuthContext = Depends(require_api_key),
) -> LongitudinalReportResponse:
    """Generate and save a timeline-ready longitudinal report for a subject."""

    return LongitudinalReportResponse(**build_saved_longitudinal_report_payload(request))
