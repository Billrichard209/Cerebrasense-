"""API request and response schemas for the backend service layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RootResponse(BaseModel):
    """Startup payload for the backend root endpoint."""

    message: str
    mode: str
    primary_dataset: str


class HealthResponse(BaseModel):
    """Health payload for local or deployed orchestration."""

    status: str
    primary_dataset: str
    decision_support_only: bool


class PolicyResponse(BaseModel):
    """Serialized safety policy for the backend."""

    product_mode: str
    diagnosis_allowed: bool
    silent_label_remap_allowed: bool
    implicit_dataset_merge_allowed: bool
    primary_dataset: str
    notes: str


class DatasetInfoResponse(BaseModel):
    """Dataset registration payload for one dataset source."""

    name: str
    priority: str
    source_root: str
    source_exists: bool
    raw_root: str
    interim_root: str
    processed_root: str
    metadata_root: str
    checkpoint_root: str
    label_column: str
    subject_id_column: str
    visit_id_column: str
    class_names: list[str] = Field(default_factory=list)


class DatasetRegistryResponse(BaseModel):
    """Full registry payload exposed by the dataset endpoint."""

    primary_dataset: str
    datasets: dict[str, DatasetInfoResponse]


class IntensityStatisticsResponse(BaseModel):
    """Serialized voxel intensity statistics."""

    minimum: float
    maximum: float
    mean: float
    std: float
    p01: float
    p50: float
    p99: float


class VolumeGeometryResponse(BaseModel):
    """Serialized 3D MRI geometry metadata."""

    shape: list[int]
    voxel_spacing_mm: list[float]
    voxel_volume_mm3: float
    affine_determinant: float


class VolumetricMeasurementResponse(BaseModel):
    """Serialized structural proxy measurement."""

    region_name: str
    value_mm3: float
    source_session: str | None = None
    unit: str
    method: str
    notes: str | None = None


class OASISVolumetricResponse(BaseModel):
    """Volumetric analysis response for one OASIS MRI volume."""

    image: str
    dataset: str
    dataset_type: str
    subject_id: str | None = None
    session_id: str | None = None
    scan_timestamp: str | None = None
    geometry: VolumeGeometryResponse
    intensity: IntensityStatisticsResponse
    foreground_intensity: IntensityStatisticsResponse | None = None
    foreground_threshold: float
    foreground_voxels: int
    nonzero_voxel_fraction: float
    bounding_box_shape_voxels: list[int]
    measurements: list[VolumetricMeasurementResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class StructuralTimepointResponse(BaseModel):
    """One structural longitudinal timepoint."""

    subject_id: str
    session_id: str | None = None
    visit_order: int
    scan_timestamp: str | None = None
    image: str
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class StructuralChangeResponse(BaseModel):
    """Change metrics for one follow-up structural timepoint."""

    session_id: str | None = None
    visit_order: int
    delta_from_baseline: dict[str, float | None] = Field(default_factory=dict)
    percent_change_from_baseline: dict[str, float | None] = Field(default_factory=dict)
    delta_from_previous: dict[str, float | None] = Field(default_factory=dict)
    percent_change_from_previous: dict[str, float | None] = Field(default_factory=dict)


class OASISStructuralLongitudinalResponse(BaseModel):
    """Subject-level OASIS structural longitudinal response."""

    subject_id: str
    dataset: str
    dataset_type: str
    timepoint_count: int
    timepoints: list[StructuralTimepointResponse] = Field(default_factory=list)
    changes: list[StructuralChangeResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModelMetadataResponse(BaseModel):
    """Metadata for the active OASIS baseline model config."""

    dataset: str
    task: str
    architecture: str
    framework: str
    class_names: list[str] = Field(default_factory=list)
    expected_input_shape: list[int] = Field(default_factory=list)
    input_shape_note: str
    densenet: dict[str, Any] = Field(default_factory=dict)
    embeddings: dict[str, Any] = Field(default_factory=dict)
    active_model_id: str | None = None
    active_checkpoint_path: str | None = None
    recommended_threshold: float | None = None
    default_threshold: float | None = None
    serving_policy: dict[str, Any] = Field(default_factory=dict)
    extension_notes: list[str] = Field(default_factory=list)
    decision_support_only: bool = True
    medical_wording: str


class ActiveModelResponse(BaseModel):
    """Approved active-model record with benchmark and promotion evidence."""

    model_id: str
    dataset: str
    run_name: str
    checkpoint_path: str
    promoted_at_utc: str
    decision_support_only: bool
    clinical_disclaimer: str
    default_threshold: float
    recommended_threshold: float
    threshold_calibration: dict[str, Any] = Field(default_factory=dict)
    temperature_scaling: dict[str, Any] = Field(default_factory=dict)
    confidence_policy: dict[str, Any] = Field(default_factory=dict)
    approval_status: str
    operational_status: str = "active"
    benchmark: dict[str, Any] = Field(default_factory=dict)
    promotion_decision: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] = Field(default_factory=dict)
    serving_restrictions: dict[str, Any] = Field(default_factory=dict)
    hold_decision: dict[str, Any] = Field(default_factory=dict)
    review_monitoring: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ReviewQueueItemResponse(BaseModel):
    """One pending human-review queue item."""

    review_id: str
    inference_id: str
    trace_id: str
    scan_id: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    source_path: str | None = None
    model_name: str
    confidence_level: str | None = None
    probability_score: float | None = None
    output_path: str | None = None
    status: str
    reason: str
    resolution: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ReviewQueueResponse(BaseModel):
    """Pending review-queue payload for low-confidence predictions."""

    total: int
    items: list[ReviewQueueItemResponse] = Field(default_factory=list)


class HoldHistoryEntryResponse(BaseModel):
    """One recorded operational-hold assessment for the active model."""

    history_path: str
    assessed_at_utc: str
    policy_name: str | None = None
    operational_status: str
    hold_applied: bool
    status_changed: bool = False
    trigger_codes: list[str] = Field(default_factory=list)
    run_name: str | None = None
    model_id: str | None = None
    summary: str | None = None
    high_risk: bool = False
    total_reviews: int = 0


class HoldHistoryResponse(BaseModel):
    """Recent operational-hold assessment history for the active model."""

    total: int
    items: list[HoldHistoryEntryResponse] = Field(default_factory=list)


class PromotionPreflightResponse(BaseModel):
    """Advisory promotion-readiness summary for one experiment candidate."""

    evaluable: bool
    approved: bool | None = None
    benchmark_name: str | None = None
    benchmark_reused_from_active: bool = False
    policy_name: str | None = None
    checks: dict[str, Any] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class PromotionCandidateResponse(BaseModel):
    """Tracked experiment surfaced as a promotion candidate."""

    experiment_name: str
    run_name: str
    experiment_root: str
    tags: list[str] = Field(default_factory=list)
    primary_split: str | None = None
    best_checkpoint_path: str | None = None
    current_active: bool = False
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] = Field(default_factory=dict)
    comparison_to_active: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    promotion_preflight: PromotionPreflightResponse
    tracked_artifacts: dict[str, str | None] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class PromotionCandidatesResponse(BaseModel):
    """List of promotion candidates from tracked experiments."""

    total: int
    items: list[PromotionCandidateResponse] = Field(default_factory=list)


class PromotionStudyResponse(BaseModel):
    """Compact model-selection study summary for promotion review."""

    study_name: str
    study_root: str
    selection_split: str
    selection_metric: str
    best_experiment_name: str | None = None
    best_run_name: str | None = None
    best_selection_score: float | None = None
    best_checkpoint_path: str | None = None
    aggregate_summary: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class PromotionHistoryDecisionResponse(BaseModel):
    """Compact summary of one saved promotion decision."""

    decision_id: str | None = None
    run_name: str
    checked_at_utc: str | None = None
    approved: bool
    benchmark_name: str | None = None
    policy_name: str | None = None
    failed_checks: list[str] = Field(default_factory=list)
    output_path: str | None = None
    history_path: str | None = None
    notes: list[str] = Field(default_factory=list)


class PromotionDashboardSummaryResponse(BaseModel):
    """Compact summary block for promotion workflow review."""

    active_run_name: str
    candidate_count: int
    promotion_ready_candidates: int
    top_candidate_experiment: str | None = None
    recommended_action: str


class PromotionDashboardResponse(BaseModel):
    """Combined promotion workflow payload for candidate-vs-active review."""

    generated_at_utc: str
    summary: PromotionDashboardSummaryResponse
    active_model: ActiveModelResponse
    candidates: PromotionCandidatesResponse
    studies: list[PromotionStudyResponse] = Field(default_factory=list)
    recent_promotion_decisions: list[PromotionHistoryDecisionResponse] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ValidationStudyResponse(BaseModel):
    """One saved validation-depth study summarized for operators."""

    study_name: str
    study_root: str
    evaluation_type: str
    run_count: int
    seed_count: int
    split_seed_count: int
    repeated_split: bool = False
    pair_seed_and_split_seed: bool = False
    active_run_included: bool = False
    active_family_included: bool = False
    selection_split: str
    selection_metric: str
    best_experiment_name: str | None = None
    best_run_name: str | None = None
    best_selection_score: float | None = None
    validation_depth_level: str
    stability_status: str
    promotion_confidence_support: str
    aggregate_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ValidationStudiesResponse(BaseModel):
    """List of validation-depth studies for the active OASIS model family."""

    total: int
    items: list[ValidationStudyResponse] = Field(default_factory=list)


class ValidationDepthSummaryResponse(BaseModel):
    """Compact summary block for validation-depth reporting."""

    active_model_id: str | None = None
    active_run_name: str
    active_run_family: str | None = None
    total_studies: int
    repeated_split_studies: int
    direct_active_run_studies: int
    related_family_studies: int
    repeated_split_family_studies: int
    overall_validation_depth: str
    recommended_action: str
    strongest_study_name: str | None = None
    strongest_stability_status: str | None = None


class ValidationDepthResponse(BaseModel):
    """Combined validation-depth dashboard for the active OASIS model family."""

    generated_at_utc: str
    summary: ValidationDepthSummaryResponse
    studies: list[ValidationStudyResponse] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReviewRiskSignalResponse(BaseModel):
    """One review-analytics warning or informational signal."""

    level: str
    code: str
    message: str
    metric: str | None = None
    value: float | int | None = None
    threshold: float | int | None = None


class ModelReviewBreakdownResponse(BaseModel):
    """Model-specific review summary within the analytics payload."""

    model_name: str
    total_reviews: int
    pending_reviews: int
    resolved_reviews: int
    adjudicated_reviews: int
    overridden_reviews: int
    confirmed_reviews: int
    dismissed_reviews: int
    override_rate: float | None = None
    confirmation_rate: float | None = None
    error_breakdown: dict[str, int] = Field(default_factory=dict)


class ReviewAnalyticsResponse(BaseModel):
    """Operational summary of human-review outcomes and model-risk signals."""

    generated_at_utc: str
    scope: str
    total_reviews: int
    pending_reviews: int
    resolved_reviews: int
    adjudicated_reviews: int
    overridden_reviews: int
    confirmed_reviews: int
    dismissed_reviews: int
    override_rate: float | None = None
    confirmation_rate: float | None = None
    status_counts: dict[str, int] = Field(default_factory=dict)
    action_counts: dict[str, int] = Field(default_factory=dict)
    reviewer_counts: dict[str, int] = Field(default_factory=dict)
    confidence_level_counts: dict[str, int] = Field(default_factory=dict)
    error_breakdown: dict[str, int] = Field(default_factory=dict)
    error_confidence_distribution: dict[str, int] = Field(default_factory=dict)
    label_override_pairs: dict[str, int] = Field(default_factory=dict)
    model_breakdown: list[ModelReviewBreakdownResponse] = Field(default_factory=list)
    average_error_probability_score: float | None = None
    high_risk: bool = False
    reviewer_agreement_available: bool = False
    reviewer_agreement_note: str
    risk_signals: list[ReviewRiskSignalResponse] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ConfidenceBandReviewResponse(BaseModel):
    """Reviewer outcome summary for one confidence band."""

    confidence_level: str
    total_cases: int
    adjudicated_cases: int
    confirmed_cases: int
    overridden_cases: int
    false_positive_count: int
    false_negative_count: int
    override_rate: float | None = None
    mean_probability_score: float | None = None


class ThresholdTuningRecommendationResponse(BaseModel):
    """Advisory threshold suggestion derived from reviewer outcomes."""

    current_threshold: float
    suggested_threshold: float
    selection_metric: str
    direction: str
    support_sample_count: int
    current_threshold_score: float | None = None
    suggested_threshold_score: float | None = None
    threshold_delta: float = 0.0
    evidence_strength: str = "insufficient"
    note: str = ""


class ReviewLearningSignalResponse(BaseModel):
    """Advisory retraining or threshold-review signal from reviewer outcomes."""

    level: str
    code: str
    message: str
    metric: str | None = None
    value: float | int | None = None
    threshold: float | int | None = None


class ReviewLearningResponse(BaseModel):
    """Advisory learning report built from reviewer-confirmed outcomes."""

    generated_at_utc: str
    scope: str
    current_threshold: float
    total_reviews: int
    resolved_reviews: int
    adjudicated_reviews: int
    confirmed_reviews: int
    overridden_reviews: int
    dismissed_reviews: int
    reviewer_labeled_samples: int
    override_rate: float | None = None
    false_positive_count: int
    false_negative_count: int
    medium_or_high_confidence_overrides: int
    recommended_action: str
    confidence_band_summary: list[ConfidenceBandReviewResponse] = Field(default_factory=list)
    threshold_recommendation: ThresholdTuningRecommendationResponse | None = None
    threshold_grid: list[dict[str, Any]] = Field(default_factory=list)
    retraining_signals: list[ReviewLearningSignalResponse] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReviewDashboardSummaryResponse(BaseModel):
    """Compact status block for reviewer operations dashboards."""

    operational_status: str
    high_risk: bool = False
    pending_reviews: int = 0
    resolved_reviews: int = 0
    hold_history_entries: int = 0
    recommended_action: str


class ReviewDashboardResponse(BaseModel):
    """Combined reviewer operations payload for queue and governance monitoring."""

    generated_at_utc: str
    summary: ReviewDashboardSummaryResponse
    active_model: ActiveModelResponse
    pending_reviews: ReviewQueueResponse
    resolved_reviews: ReviewQueueResponse
    analytics: ReviewAnalyticsResponse
    hold_history: HoldHistoryResponse
    notes: list[str] = Field(default_factory=list)


class ReviewResolutionRequest(BaseModel):
    """Reviewer action payload for resolving one queued case."""

    reviewer_id: str = Field(min_length=1, max_length=128)
    action: Literal["confirm_model_output", "override_prediction", "dismiss"]
    resolution_note: str | None = Field(default=None, max_length=2000)
    resolved_label: int | None = None
    resolved_label_name: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _validate_resolution(self) -> "ReviewResolutionRequest":
        """Ensure override actions carry an explicit final label."""

        if self.action == "override_prediction" and self.resolved_label is None:
            raise ValueError("resolved_label is required when action='override_prediction'.")
        if self.action != "override_prediction" and self.resolved_label is not None:
            raise ValueError("resolved_label is only valid when action='override_prediction'.")
        return self


class ReviewResolutionResponse(BaseModel):
    """Updated review item returned after a reviewer action."""

    review_id: str
    status: str
    message: str
    item: ReviewQueueItemResponse


class ScanPredictionRequest(BaseModel):
    """Request to run inference for one existing scan path."""

    scan_path: str
    checkpoint_path: str
    config_path: str | None = None
    model_config_path: str | None = None
    output_name: str = "api_scan_prediction"
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    device: str = "cpu"
    save_debug_slices: bool = False
    subject_id: str | None = None
    session_id: str | None = None
    scan_timestamp: str | None = None


class ScanPredictionResponse(BaseModel):
    """Decision-support prediction payload for one scan."""

    prediction_id: str | None = None
    trace_id: str | None = None
    predicted_label: int
    label_name: str
    probability_score: float
    calibrated_probability_score: float | None = None
    confidence_score: float
    confidence_level: str | None = None
    review_flag: bool = False
    risk_category: str | None = None
    model_name: str
    preprocessing_config: str
    checkpoint_path: str | None = None
    active_model_id: str | None = None
    operational_status: str | None = None
    serving_restrictions: dict[str, Any] = Field(default_factory=dict)
    input_metadata: dict[str, Any] = Field(default_factory=dict)
    ai_summary: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    decision_support_only: bool
    clinical_disclaimer: str | None = None
    abnormal_regions: list[Any] = Field(default_factory=list)
    heatmap_visualization: str | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ScanExplanationRequest(BaseModel):
    """Request to generate Grad-CAM-style explanation artifacts for one scan."""

    scan_path: str
    checkpoint_path: str
    preprocessing_config_path: str | None = None
    model_config_path: str | None = None
    output_name: str = "api_scan_explanation"
    target_layer: str = "auto"
    target_class: int | None = None
    device: str = "cpu"
    image_size: list[int] | None = None
    slice_axis: str = "axial"
    slice_indices: list[int] | None = None
    save_saliency: bool = True
    true_label: int | None = None

    @field_validator("image_size")
    @classmethod
    def _validate_image_size(cls, value: list[int] | None) -> list[int] | None:
        """Ensure image-size overrides remain 3D."""

        if value is not None and len(value) != 3:
            raise ValueError("image_size must contain exactly three integers: D, H, W.")
        return value


class ScanExplanationResponse(BaseModel):
    """Saved explainability artifact response for one scan."""

    method: str
    dataset_assumption: str
    scan_path: str
    checkpoint_path: str
    preprocessing_config: str
    model_name: str
    target_layer: str
    target_class: int
    target_class_name: str
    true_label: int | None = None
    prediction_correctness: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    confidence_level: str | None = None
    review_flag: bool = False
    region_importance_proxy: dict[str, float] = Field(default_factory=dict)
    heatmap_intensity_summary: dict[str, float] = Field(default_factory=dict)
    highlighted_regions: str | None = None
    confidence_interpretation: str | None = None
    explanation_quality: str | None = None
    decision_support_only: bool = True
    clinical_disclaimer: str | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class APITrendFeatureConfig(BaseModel):
    """Configurable trend feature thresholds for longitudinal reporting."""

    feature_name: str
    source: str
    decline_direction: str
    normalization: str = "percent_from_baseline"
    stable_slope_threshold: float = Field(default=1.0, ge=0.0)
    rapid_slope_threshold: float = Field(default=5.0, ge=0.0)
    display_name: str | None = None
    unit: str | None = None
    notes: str | None = None


class APIScanHistoryRecord(BaseModel):
    """One scan-history timepoint for API longitudinal reports."""

    subject_id: str
    session_id: str | None = None
    visit_order: int | None = None
    summary_label: str | None = None
    scan_timestamp: str | None = None
    source_path: str | None = None
    dataset: str | None = None
    volumetric_features: dict[str, float] = Field(default_factory=dict)
    model_probabilities: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LongitudinalReportRequest(BaseModel):
    """Request to generate and save a subject-level longitudinal report.

    If ``records`` are supplied, the general timeline/trend engine is used. If
    no records are supplied, the endpoint keeps the existing OASIS structural
    proxy behavior using ``subject_id`` plus optional manifest fields.
    """

    subject_id: str
    records: list[APIScanHistoryRecord] = Field(default_factory=list)
    feature_configs: list[APITrendFeatureConfig] = Field(default_factory=list)
    split: str | None = None
    manifest_path: str | None = None
    max_timepoints: int | None = Field(default=None, ge=1)
    output_name: str | None = None


class LongitudinalReportResponse(BaseModel):
    """Saved longitudinal report payload with timeline or structural metadata."""

    subject_id: str
    report_type: str
    output_path: str
    timepoint_count: int
    generated_at: str | None = None
    timepoints: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    interval_changes: list[dict[str, Any]] = Field(default_factory=list)
    trend_summaries: list[dict[str, Any]] = Field(default_factory=list)
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    progression_overview: dict[str, Any] = Field(default_factory=dict)
    feature_configs: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    dataset: str | None = None
    dataset_type: str | None = None
    changes: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
