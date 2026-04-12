"""Governance and review-operations endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.auth import AuthContext, require_api_key
from src.api.schemas import (
    ActiveModelResponse,
    HoldHistoryResponse,
    PromotionCandidatesResponse,
    PromotionDashboardResponse,
    ReviewAnalyticsResponse,
    ReviewDashboardResponse,
    ReviewLearningResponse,
    ReviewQueueItemResponse,
    ReviewQueueResponse,
    ReviewResolutionRequest,
    ReviewResolutionResponse,
    ValidationDepthResponse,
    ValidationStudiesResponse,
)
from src.api.services import (
    build_active_oasis_model_payload,
    build_hold_history_payload,
    build_promotion_candidates_payload,
    build_promotion_dashboard_payload,
    build_review_detail_payload,
    build_resolved_review_queue_payload,
    build_review_analytics_payload,
    build_review_dashboard_payload,
    build_review_learning_payload,
    build_validation_depth_payload,
    build_validation_studies_payload,
    build_pending_review_queue_payload,
    resolve_review_queue_item_payload,
)

router = APIRouter(tags=["governance"])


@router.get("/models/oasis/active", response_model=ActiveModelResponse)
def active_oasis_model() -> ActiveModelResponse:
    """Return the active OASIS model plus benchmark-backed approval evidence."""

    return ActiveModelResponse(**build_active_oasis_model_payload())


@router.get("/reviews/pending", response_model=ReviewQueueResponse)
def pending_reviews(
    limit: int = Query(default=20, ge=1, le=200),
    _auth: AuthContext = Depends(require_api_key),
) -> ReviewQueueResponse:
    """Return pending low-confidence review cases for manual follow-up."""

    return ReviewQueueResponse(**build_pending_review_queue_payload(limit=limit))


@router.get("/reviews/analytics", response_model=ReviewAnalyticsResponse)
def review_analytics(
    limit: int = Query(default=200, ge=1, le=1000),
    model_name: str | None = Query(default=None),
    active_model_only: bool = Query(default=False),
    _auth: AuthContext = Depends(require_api_key),
) -> ReviewAnalyticsResponse:
    """Return aggregate review outcomes plus post-promotion warning signals."""

    return ReviewAnalyticsResponse(
        **build_review_analytics_payload(
            limit=limit,
            model_name=model_name,
            active_model_only=active_model_only,
        )
    )


@router.get("/reviews/learning-report", response_model=ReviewLearningResponse)
def review_learning_report(
    limit: int = Query(default=200, ge=1, le=1000),
    model_name: str | None = Query(default=None),
    active_model_only: bool = Query(default=False),
    selection_metric: str = Query(default="balanced_accuracy"),
    threshold_step: float = Query(default=0.05, gt=0.0, le=0.5),
    _auth: AuthContext = Depends(require_api_key),
) -> ReviewLearningResponse:
    """Return an advisory reviewer-outcome learning report for threshold and retraining review."""

    try:
        return ReviewLearningResponse(
            **build_review_learning_payload(
                limit=limit,
                model_name=model_name,
                active_model_only=active_model_only,
                selection_metric=selection_metric,
                threshold_step=threshold_step,
            )
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.get("/reviews/resolved", response_model=ReviewQueueResponse)
def resolved_reviews(
    limit: int = Query(default=20, ge=1, le=200),
    status_filter: str | None = Query(default=None, alias="status"),
    _auth: AuthContext = Depends(require_api_key),
) -> ReviewQueueResponse:
    """Return recently resolved review cases for reviewer follow-up."""

    try:
        return ReviewQueueResponse(
            **build_resolved_review_queue_payload(
                limit=limit,
                status=status_filter,
            )
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.get("/models/oasis/hold-history", response_model=HoldHistoryResponse)
def active_oasis_hold_history(
    limit: int = Query(default=20, ge=1, le=200),
    _auth: AuthContext = Depends(require_api_key),
) -> HoldHistoryResponse:
    """Return recent operational-hold assessments for the active OASIS model."""

    return HoldHistoryResponse(**build_hold_history_payload(limit=limit))


@router.get("/models/oasis/promotion-candidates", response_model=PromotionCandidatesResponse)
def promotion_candidates(
    limit: int = Query(default=10, ge=1, le=100),
    _auth: AuthContext = Depends(require_api_key),
) -> PromotionCandidatesResponse:
    """Return tracked OASIS experiments surfaced as promotion candidates."""

    return PromotionCandidatesResponse(**build_promotion_candidates_payload(limit=limit))


@router.get("/models/oasis/promotion-dashboard", response_model=PromotionDashboardResponse)
def promotion_dashboard(
    candidate_limit: int = Query(default=5, ge=1, le=50),
    study_limit: int = Query(default=5, ge=1, le=50),
    history_limit: int = Query(default=5, ge=1, le=50),
    _auth: AuthContext = Depends(require_api_key),
) -> PromotionDashboardResponse:
    """Return a compact candidate-vs-active promotion review payload."""

    return PromotionDashboardResponse(
        **build_promotion_dashboard_payload(
            candidate_limit=candidate_limit,
            study_limit=study_limit,
            history_limit=history_limit,
        )
    )


@router.get("/models/oasis/validation-studies", response_model=ValidationStudiesResponse)
def validation_studies(
    limit: int = Query(default=10, ge=1, le=100),
    _auth: AuthContext = Depends(require_api_key),
) -> ValidationStudiesResponse:
    """Return saved repeated-split and multi-seed validation studies for the active model family."""

    return ValidationStudiesResponse(**build_validation_studies_payload(limit=limit))


@router.get("/models/oasis/validation-depth", response_model=ValidationDepthResponse)
def validation_depth(
    limit: int = Query(default=10, ge=1, le=100),
    _auth: AuthContext = Depends(require_api_key),
) -> ValidationDepthResponse:
    """Return a compact validation-depth dashboard for the active OASIS model family."""

    return ValidationDepthResponse(**build_validation_depth_payload(limit=limit))


@router.get("/reviews/dashboard", response_model=ReviewDashboardResponse)
def review_dashboard(
    pending_limit: int = Query(default=10, ge=1, le=100),
    resolved_limit: int = Query(default=10, ge=1, le=100),
    history_limit: int = Query(default=10, ge=1, le=100),
    _auth: AuthContext = Depends(require_api_key),
) -> ReviewDashboardResponse:
    """Return a compact reviewer-operations dashboard payload."""

    return ReviewDashboardResponse(
        **build_review_dashboard_payload(
            pending_limit=pending_limit,
            resolved_limit=resolved_limit,
            history_limit=history_limit,
        )
    )


@router.get("/reviews/{review_id}", response_model=ReviewQueueItemResponse)
def review_detail(
    review_id: str,
    _auth: AuthContext = Depends(require_api_key),
) -> ReviewQueueItemResponse:
    """Return one queued or resolved review case by id."""

    try:
        return ReviewQueueItemResponse(**build_review_detail_payload(review_id))
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.post("/reviews/{review_id}/resolve", response_model=ReviewResolutionResponse)
def resolve_review(
    review_id: str,
    request: ReviewResolutionRequest,
    _auth: AuthContext = Depends(require_api_key),
) -> ReviewResolutionResponse:
    """Resolve one pending low-confidence review case."""

    try:
        return ReviewResolutionResponse(**resolve_review_queue_item_payload(review_id, request))
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
