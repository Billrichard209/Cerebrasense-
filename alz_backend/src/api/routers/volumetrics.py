"""Volumetric analysis API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.auth import AuthContext, require_api_key
from src.api.schemas import OASISVolumetricResponse
from src.api.services import build_oasis_volumetric_payload

router = APIRouter(tags=["volumetrics"])


@router.get("/volumetrics/oasis", response_model=OASISVolumetricResponse)
def oasis_volumetrics(
    image_path: str | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    scan_timestamp: str | None = None,
    split: str | None = None,
    row_index: int = 0,
    manifest_path: str | None = None,
    _auth: AuthContext = Depends(require_api_key),
) -> OASISVolumetricResponse:
    """Run a foreground-proxy structural analysis for one OASIS MRI volume."""

    return OASISVolumetricResponse(
        **build_oasis_volumetric_payload(
            image_path=image_path,
            subject_id=subject_id,
            session_id=session_id,
            scan_timestamp=scan_timestamp,
            split=split,
            row_index=row_index,
            manifest_path=manifest_path,
        )
    )
