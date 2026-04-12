"""Explainability API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.auth import AuthContext, require_api_key
from src.api.schemas import ScanExplanationRequest, ScanExplanationResponse
from src.api.services import build_scan_explanation_payload

router = APIRouter(tags=["explainability"])


@router.post("/explain/scan", response_model=ScanExplanationResponse)
def explain_scan_route(
    request: ScanExplanationRequest,
    _auth: AuthContext = Depends(require_api_key),
) -> ScanExplanationResponse:
    """Generate Grad-CAM-style explanation artifacts for one scan."""

    return ScanExplanationResponse(**build_scan_explanation_payload(request))

