"""Scan-level inference API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query

from src.api.auth import AuthContext, require_api_key
from src.api.schemas import ScanPredictionRequest, ScanPredictionResponse
from src.api.services import build_scan_prediction_payload, build_scan_prediction_upload_payload

router = APIRouter(tags=["inference"])


@router.post("/predict/scan", response_model=ScanPredictionResponse)
def predict_scan_route(
    request: ScanPredictionRequest,
    _auth: AuthContext = Depends(require_api_key),
) -> ScanPredictionResponse:
    """Predict a single existing scan path using a trained checkpoint."""

    return ScanPredictionResponse(**build_scan_prediction_payload(request))


@router.post("/predict/scan/upload", response_model=ScanPredictionResponse)
def predict_scan_upload_route(
    scan_bytes: Annotated[bytes, Body(media_type="application/octet-stream")],
    file_name: Annotated[str, Query(description="Original scan file name. NIfTI uploads are supported here.")],
    checkpoint_path: Annotated[str, Query(description="Path to a trained OASIS checkpoint.")],
    config_path: str | None = None,
    model_config_path: str | None = None,
    output_name: str = "api_upload_prediction",
    threshold: float | None = None,
    device: str = "cpu",
    save_debug_slices: bool = False,
    subject_id: str | None = None,
    session_id: str | None = None,
    scan_timestamp: str | None = None,
    _auth: AuthContext = Depends(require_api_key),
) -> ScanPredictionResponse:
    """Predict a raw binary NIfTI upload without relying on multipart parsing."""

    return ScanPredictionResponse(
        **build_scan_prediction_upload_payload(
            file_name=file_name,
            content=scan_bytes,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            model_config_path=model_config_path,
            output_name=output_name,
            threshold=threshold,
            device=device,
            save_debug_slices=save_debug_slices,
            subject_id=subject_id,
            session_id=session_id,
            scan_timestamp=scan_timestamp,
        )
    )
