"""Tests for the modular FastAPI backend layer."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app


def _prediction_payload() -> dict[str, object]:
    """Return a minimal prediction payload matching the API response schema."""

    return {
        "predicted_label": 1,
        "label_name": "demented",
        "probability_score": 0.67,
        "calibrated_probability_score": 0.66,
        "confidence_score": 0.67,
        "confidence_level": "medium",
        "review_flag": False,
        "risk_category": "demented",
        "model_name": "densenet121_3d",
        "preprocessing_config": "configs/oasis_transforms.yaml",
        "input_metadata": {"scan_path": "scan.nii.gz", "dataset_assumption": "oasis1_3d_volume"},
        "ai_summary": "Decision-support model output; not a diagnosis.",
        "probabilities": {"nondemented": 0.33, "demented": 0.67},
        "uncertainty": {"confidence": 0.67},
        "decision_support_only": True,
        "clinical_disclaimer": "This output is for research and clinical decision support only. It is not a diagnosis.",
        "abnormal_regions": [],
        "heatmap_visualization": None,
        "outputs": {"prediction_json": "prediction.json"},
        "notes": ["Decision-support only."],
    }


def _explanation_payload() -> dict[str, object]:
    """Return a minimal explanation payload matching the API response schema."""

    return {
        "method": "grad_cam_style_3d",
        "dataset_assumption": "oasis1_3d_volume",
        "scan_path": "scan.nii.gz",
        "checkpoint_path": "checkpoint.pt",
        "preprocessing_config": "configs/oasis_transforms.yaml",
        "model_name": "densenet121_3d",
        "target_layer": "features.denseblock4",
        "target_class": 1,
        "target_class_name": "demented",
        "true_label": 1,
        "prediction_correctness": "correct",
        "probabilities": {"nondemented": 0.33, "demented": 0.67},
        "uncertainty": {"confidence": 0.67},
        "confidence_level": "medium",
        "review_flag": False,
        "region_importance_proxy": {"first_depth_half_proxy": 0.25},
        "heatmap_intensity_summary": {"max": 0.8},
        "highlighted_regions": "Highest coarse heatmap activity appeared in: first_depth_half_proxy.",
        "confidence_interpretation": "The model response was moderately concentrated.",
        "explanation_quality": "uncertain",
        "decision_support_only": True,
        "clinical_disclaimer": "This output is for research and clinical decision support only. It is not a diagnosis.",
        "artifacts": {"report_json": "explanation_report.json"},
        "limitations": ["Grad-CAM is not a diagnosis."],
    }


def test_root_endpoint_uses_structural_mri_wording() -> None:
    """The root endpoint should use the narrowed structural-MRI framing."""

    client = TestClient(create_app())
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "decision_support"
    assert "structural mri backend core" in payload["message"].lower()


def test_model_metadata_endpoint_returns_decision_support_metadata() -> None:
    """Model metadata should be available without running inference."""

    client = TestClient(create_app())
    response = client.get("/models/oasis/metadata")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == "oasis1"
    assert payload["framework"] == "monai"
    assert payload["decision_support_only"] is True


def test_active_model_endpoint_returns_governance_payload(monkeypatch) -> None:
    """The active-model endpoint should expose benchmark and approval evidence."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_active_oasis_model_payload",
        lambda: {
            "model_id": "oasis_current_baseline",
            "dataset": "oasis1",
            "run_name": "unit_run",
            "checkpoint_path": "checkpoint.pt",
            "promoted_at_utc": "2026-01-01T00:00:00+00:00",
            "decision_support_only": True,
            "clinical_disclaimer": "Decision-support only.",
            "default_threshold": 0.5,
            "recommended_threshold": 0.42,
            "threshold_calibration": {"threshold": 0.42, "selection_metric": "balanced_accuracy"},
            "temperature_scaling": {"enabled": False, "temperature": 1.0},
            "confidence_policy": {"high_confidence_min": 0.85},
            "approval_status": "approved",
            "benchmark": {"benchmark_name": "oasis_test_split42"},
            "promotion_decision": {"approved": True},
            "evidence": {"test_metrics_path": "metrics.json"},
            "validation_metrics": {"auroc": 0.8},
            "test_metrics": {"auroc": 0.84},
            "notes": ["Decision-support only."],
        },
    )
    client = TestClient(create_app())
    response = client.get("/models/oasis/active")

    assert response.status_code == 200
    payload = response.json()
    assert payload["approval_status"] == "approved"
    assert payload["benchmark"]["benchmark_name"] == "oasis_test_split42"


def test_pending_reviews_endpoint_returns_service_payload(monkeypatch) -> None:
    """The review queue endpoint should return pending review items."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_pending_review_queue_payload",
        lambda limit=20: {
            "total": 1,
            "items": [
                {
                    "review_id": "review-1",
                    "inference_id": "inf-1",
                    "trace_id": "trace-1",
                    "scan_id": "scan-1",
                    "subject_id": "OAS1_0001",
                    "session_id": None,
                    "source_path": "scan.nii.gz",
                    "model_name": "densenet121_3d",
                    "confidence_level": "low",
                    "probability_score": 0.54,
                    "output_path": "prediction.json",
                    "status": "pending",
                    "reason": "low_confidence_prediction",
                    "payload": {"review_flag": True},
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ],
        },
    )
    client = TestClient(create_app())
    response = client.get("/reviews/pending")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["reason"] == "low_confidence_prediction"


def test_review_detail_endpoint_returns_service_payload(monkeypatch) -> None:
    """The review-detail route should return one normalized review record."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_review_detail_payload",
        lambda review_id: {
            "review_id": review_id,
            "inference_id": "inf-1",
            "trace_id": "trace-1",
            "scan_id": "scan-1",
            "subject_id": "OAS1_0001",
            "session_id": None,
            "source_path": "scan.nii.gz",
            "model_name": "densenet121_3d",
            "confidence_level": "low",
            "probability_score": 0.54,
            "output_path": "prediction.json",
            "status": "pending",
            "reason": "low_confidence_prediction",
            "resolution": {},
            "payload": {"review_flag": True},
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )
    client = TestClient(create_app())
    response = client.get("/reviews/review-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["review_id"] == "review-1"
    assert payload["status"] == "pending"


def test_review_analytics_endpoint_returns_service_payload(monkeypatch) -> None:
    """The review analytics route should expose aggregate review monitoring."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_review_analytics_payload",
        lambda limit=200, model_name=None, active_model_only=False: {
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "scope": "active_model_id=oasis_current_baseline, model_name=densenet121_3d",
            "total_reviews": 6,
            "pending_reviews": 1,
            "resolved_reviews": 5,
            "adjudicated_reviews": 4,
            "overridden_reviews": 2,
            "confirmed_reviews": 2,
            "dismissed_reviews": 1,
            "override_rate": 0.5,
            "confirmation_rate": 0.5,
            "status_counts": {"overridden": 2, "confirmed": 2, "pending": 1, "dismissed": 1},
            "action_counts": {"override_prediction": 2, "confirm_model_output": 2, "dismiss": 1},
            "reviewer_counts": {"reviewer_a": 3, "reviewer_b": 2},
            "confidence_level_counts": {"low": 4, "medium": 2},
            "error_breakdown": {"false_positive": 1, "false_negative": 1},
            "error_confidence_distribution": {"low": 2},
            "label_override_pairs": {"demented -> nondemented": 1, "nondemented -> demented": 1},
            "model_breakdown": [
                {
                    "model_name": "densenet121_3d",
                    "total_reviews": 6,
                    "pending_reviews": 1,
                    "resolved_reviews": 5,
                    "adjudicated_reviews": 4,
                    "overridden_reviews": 2,
                    "confirmed_reviews": 2,
                    "dismissed_reviews": 1,
                    "override_rate": 0.5,
                    "confirmation_rate": 0.5,
                    "error_breakdown": {"false_positive": 1, "false_negative": 1},
                }
            ],
            "average_error_probability_score": 0.51,
            "high_risk": True,
            "reviewer_agreement_available": False,
            "reviewer_agreement_note": "Single-review resolution workflow only.",
            "risk_signals": [
                {
                    "level": "warning",
                    "code": "high_override_rate",
                    "message": "Override rate is high.",
                    "metric": "override_rate",
                    "value": 0.5,
                    "threshold": 0.3,
                }
            ],
            "notes": ["Post-promotion review monitoring is active."],
        },
    )
    client = TestClient(create_app())
    response = client.get("/reviews/analytics?active_model_only=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["high_risk"] is True
    assert payload["risk_signals"][0]["code"] == "high_override_rate"
    assert payload["model_breakdown"][0]["model_name"] == "densenet121_3d"


def test_review_learning_endpoint_returns_service_payload(monkeypatch) -> None:
    """The review learning route should expose threshold and retraining guidance."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_review_learning_payload",
        lambda limit=200, model_name=None, active_model_only=False, selection_metric="balanced_accuracy", threshold_step=0.05: {
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "scope": "active_model_id=oasis_current_baseline, run_name=run42, model_name=densenet121_3d",
            "current_threshold": 0.45,
            "total_reviews": 8,
            "resolved_reviews": 7,
            "adjudicated_reviews": 7,
            "confirmed_reviews": 4,
            "overridden_reviews": 3,
            "dismissed_reviews": 0,
            "reviewer_labeled_samples": 7,
            "override_rate": 0.4285714286,
            "false_positive_count": 3,
            "false_negative_count": 0,
            "medium_or_high_confidence_overrides": 3,
            "recommended_action": "Run a validation-only threshold recalibration experiment using the reviewer-guided suggestion, then compare against the current approved setting.",
            "confidence_band_summary": [
                {
                    "confidence_level": "medium",
                    "total_cases": 4,
                    "adjudicated_cases": 4,
                    "confirmed_cases": 2,
                    "overridden_cases": 2,
                    "false_positive_count": 2,
                    "false_negative_count": 0,
                    "override_rate": 0.5,
                    "mean_probability_score": 0.495,
                }
            ],
            "threshold_recommendation": {
                "current_threshold": 0.45,
                "suggested_threshold": 0.6,
                "selection_metric": "balanced_accuracy",
                "direction": "raise_threshold",
                "support_sample_count": 7,
                "current_threshold_score": 0.7,
                "suggested_threshold_score": 1.0,
                "threshold_delta": 0.15,
                "evidence_strength": "limited",
                "note": "Advisory only.",
            },
            "threshold_grid": [{"threshold": 0.45, "selection_score": 0.7}],
            "retraining_signals": [
                {
                    "level": "warning",
                    "code": "false_positive_pattern",
                    "message": "Reviewer outcomes lean toward false positives.",
                    "metric": "false_positive_count",
                    "value": 3,
                    "threshold": 3,
                }
            ],
            "notes": ["Reviewer feedback is selection-biased toward queued cases."],
        },
    )
    client = TestClient(create_app())
    response = client.get("/reviews/learning-report?active_model_only=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_threshold"] == 0.45
    assert payload["threshold_recommendation"]["direction"] == "raise_threshold"
    assert payload["retraining_signals"][0]["code"] == "false_positive_pattern"


def test_resolved_reviews_endpoint_returns_service_payload(monkeypatch) -> None:
    """The resolved review endpoint should expose reviewed cases."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_resolved_review_queue_payload",
        lambda limit=20, status=None: {
            "total": 1,
            "items": [
                {
                    "review_id": "review-2",
                    "inference_id": "inf-2",
                    "trace_id": "trace-2",
                    "scan_id": "scan-2",
                    "subject_id": "OAS1_0002",
                    "session_id": None,
                    "source_path": "scan2.nii.gz",
                    "model_name": "densenet121_3d",
                    "confidence_level": "low",
                    "probability_score": 0.53,
                    "output_path": "prediction2.json",
                    "status": "confirmed",
                    "reason": "low_confidence_prediction",
                    "resolution": {"reviewer_id": "reviewer_a"},
                    "payload": {"review_flag": False},
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ],
        },
    )
    client = TestClient(create_app())
    response = client.get("/reviews/resolved")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["status"] == "confirmed"


def test_hold_history_endpoint_returns_service_payload(monkeypatch) -> None:
    """The hold-history endpoint should expose recent operational assessments."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_hold_history_payload",
        lambda limit=20: {
            "total": 1,
            "items": [
                {
                    "history_path": "hold_history.json",
                    "assessed_at_utc": "2026-01-01T00:00:00+00:00",
                    "policy_name": "oasis_operational_hold_v1",
                    "operational_status": "active",
                    "hold_applied": False,
                    "status_changed": False,
                    "trigger_codes": [],
                    "run_name": "unit_run",
                    "model_id": "oasis_current_baseline",
                    "summary": "No hold applied.",
                    "high_risk": False,
                    "total_reviews": 3,
                }
            ],
        },
    )
    client = TestClient(create_app())
    response = client.get("/models/oasis/hold-history")

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["operational_status"] == "active"
    assert payload["items"][0]["hold_applied"] is False


def test_promotion_candidates_endpoint_returns_service_payload(monkeypatch) -> None:
    """The promotion-candidates endpoint should expose tracked experiment candidates."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_promotion_candidates_payload",
        lambda limit=10: {
            "total": 1,
            "items": [
                {
                    "experiment_name": "candidate_a",
                    "run_name": "run_a",
                    "experiment_root": "outputs/experiments/candidate_a",
                    "tags": ["baseline"],
                    "primary_split": "val",
                    "best_checkpoint_path": "checkpoint.pt",
                    "current_active": False,
                    "validation_metrics": {"auroc": 0.82},
                    "test_metrics": {"auroc": 0.84},
                    "comparison_to_active": {"test": {"auroc": 0.02}},
                    "promotion_preflight": {
                        "evaluable": True,
                        "approved": True,
                        "benchmark_name": "oasis1_test_split42_frozen",
                        "benchmark_reused_from_active": True,
                        "policy_name": "oasis_research_gate_v1",
                        "checks": {},
                        "failed_checks": [],
                        "evidence_summary": {"test_auroc": 0.84},
                        "notes": [],
                    },
                    "tracked_artifacts": {"summary_path": "experiment_summary.json"},
                    "notes": ["advisory only"],
                }
            ],
        },
    )
    client = TestClient(create_app())
    response = client.get("/models/oasis/promotion-candidates")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["promotion_preflight"]["approved"] is True


def test_promotion_dashboard_endpoint_returns_service_payload(monkeypatch) -> None:
    """The promotion-dashboard endpoint should aggregate active-vs-candidate review state."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_promotion_dashboard_payload",
        lambda candidate_limit=5, study_limit=5, history_limit=5: {
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "summary": {
                "active_run_name": "active_run",
                "candidate_count": 2,
                "promotion_ready_candidates": 1,
                "top_candidate_experiment": "candidate_a",
                "recommended_action": "Candidate 'candidate_a' is preflight-ready.",
            },
            "active_model": {
                "model_id": "oasis_current_baseline",
                "dataset": "oasis1",
                "run_name": "active_run",
                "checkpoint_path": "checkpoint.pt",
                "promoted_at_utc": "2026-01-01T00:00:00+00:00",
                "decision_support_only": True,
                "clinical_disclaimer": "Decision-support only.",
                "default_threshold": 0.5,
                "recommended_threshold": 0.45,
                "threshold_calibration": {},
                "temperature_scaling": {},
                "confidence_policy": {},
                "approval_status": "approved",
                "operational_status": "active",
                "benchmark": {},
                "promotion_decision": {},
                "evidence": {},
                "validation_metrics": {},
                "test_metrics": {},
                "serving_restrictions": {},
                "hold_decision": {},
                "review_monitoring": {},
                "notes": [],
            },
            "candidates": {
                "total": 2,
                "items": [],
            },
            "studies": [
                {
                    "study_name": "seed_sweep",
                    "study_root": "outputs/model_selection/seed_sweep",
                    "selection_split": "val",
                    "selection_metric": "auroc",
                    "best_experiment_name": "candidate_a",
                    "best_run_name": "run_a",
                    "best_selection_score": 0.9,
                    "best_checkpoint_path": "checkpoint.pt",
                    "aggregate_summary": {"test_auroc_mean": 0.8},
                    "notes": [],
                }
            ],
            "recent_promotion_decisions": [
                {
                    "decision_id": "decision-1",
                    "run_name": "active_run",
                    "checked_at_utc": "2026-01-01T00:00:00+00:00",
                    "approved": True,
                    "benchmark_name": "oasis1_test_split42_frozen",
                    "policy_name": "oasis_research_gate_v1",
                    "failed_checks": [],
                    "output_path": "registry.json",
                    "history_path": "promotion_history.json",
                    "notes": [],
                }
            ],
            "notes": ["advisory only"],
        },
    )
    client = TestClient(create_app())
    response = client.get("/models/oasis/promotion-dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["candidate_count"] == 2
    assert payload["studies"][0]["study_name"] == "seed_sweep"
    assert payload["recent_promotion_decisions"][0]["approved"] is True


def test_validation_studies_endpoint_returns_service_payload(monkeypatch) -> None:
    """The validation-studies route should expose saved study summaries."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_validation_studies_payload",
        lambda limit=10: {
            "total": 2,
            "items": [
                {
                    "study_name": "fixed_split_family",
                    "study_root": "outputs/model_selection/fixed_split_family",
                    "evaluation_type": "multi_seed_fixed_split",
                    "run_count": 3,
                    "seed_count": 3,
                    "split_seed_count": 1,
                    "repeated_split": False,
                    "pair_seed_and_split_seed": False,
                    "active_run_included": True,
                    "active_family_included": True,
                    "selection_split": "val",
                    "selection_metric": "auroc",
                    "best_experiment_name": "fixed_seed42",
                    "best_run_name": "oasis_baseline_rtx2050_gpu_seed42_split42",
                    "best_selection_score": 0.81,
                    "validation_depth_level": "moderate",
                    "stability_status": "moderate",
                    "promotion_confidence_support": "moderate",
                    "aggregate_summary": {"test_auroc": {"mean": 0.77, "std": 0.04}},
                    "warnings": [],
                    "notes": ["Fixed split coverage for the active family."],
                },
                {
                    "study_name": "repeated_split_smoke",
                    "study_root": "outputs/model_selection/repeated_split_smoke",
                    "evaluation_type": "repeated_subject_safe_splits",
                    "run_count": 6,
                    "seed_count": 3,
                    "split_seed_count": 2,
                    "repeated_split": True,
                    "pair_seed_and_split_seed": False,
                    "active_run_included": False,
                    "active_family_included": True,
                    "selection_split": "val",
                    "selection_metric": "auroc",
                    "best_experiment_name": "smoke_seed42_split101",
                    "best_run_name": "oasis_baseline_rtx2050_gpu_seed42_split101",
                    "best_selection_score": 0.0,
                    "validation_depth_level": "insufficient",
                    "stability_status": "insufficient",
                    "promotion_confidence_support": "insufficient",
                    "aggregate_summary": {"test_sample_count": {"mean": 2.0, "std": 0.0}},
                    "warnings": ["Held-out sample count is too small for trustworthy stability claims."],
                    "notes": ["Tiny repeated-split smoke evidence only."],
                },
            ],
        },
    )
    client = TestClient(create_app())
    response = client.get("/models/oasis/validation-studies")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["items"][0]["evaluation_type"] == "multi_seed_fixed_split"
    assert payload["items"][1]["repeated_split"] is True


def test_validation_depth_endpoint_returns_service_payload(monkeypatch) -> None:
    """The validation-depth route should expose the family-level stability summary."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_validation_depth_payload",
        lambda limit=10: {
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "summary": {
                "active_model_id": "oasis_current_baseline",
                "active_run_name": "oasis_baseline_rtx2050_gpu_seed42_split42",
                "active_run_family": "oasis_baseline_rtx2050_gpu",
                "total_studies": 2,
                "repeated_split_studies": 1,
                "direct_active_run_studies": 1,
                "related_family_studies": 2,
                "repeated_split_family_studies": 1,
                "overall_validation_depth": "moderate",
                "recommended_action": "Validation depth is improving, but repeat subject-safe studies and external validation should stay high priority.",
                "strongest_study_name": "fixed_split_family",
                "strongest_stability_status": "moderate",
            },
            "studies": [
                {
                    "study_name": "fixed_split_family",
                    "study_root": "outputs/model_selection/fixed_split_family",
                    "evaluation_type": "multi_seed_fixed_split",
                    "run_count": 3,
                    "seed_count": 3,
                    "split_seed_count": 1,
                    "repeated_split": False,
                    "pair_seed_and_split_seed": False,
                    "active_run_included": True,
                    "active_family_included": True,
                    "selection_split": "val",
                    "selection_metric": "auroc",
                    "best_experiment_name": "fixed_seed42",
                    "best_run_name": "oasis_baseline_rtx2050_gpu_seed42_split42",
                    "best_selection_score": 0.81,
                    "validation_depth_level": "moderate",
                    "stability_status": "moderate",
                    "promotion_confidence_support": "moderate",
                    "aggregate_summary": {"test_auroc": {"mean": 0.77, "std": 0.04}},
                    "warnings": [],
                    "notes": [],
                }
            ],
            "notes": [
                "Validation depth is about stability and generalization evidence, not just one good metric snapshot.",
            ],
        },
    )
    client = TestClient(create_app())
    response = client.get("/models/oasis/validation-depth")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["overall_validation_depth"] == "moderate"
    assert payload["summary"]["repeated_split_studies"] == 1
    assert payload["studies"][0]["study_name"] == "fixed_split_family"


def test_review_dashboard_endpoint_returns_service_payload(monkeypatch) -> None:
    """The dashboard endpoint should aggregate reviewer operations state."""

    monkeypatch.setattr(
        "src.api.routers.governance.build_review_dashboard_payload",
        lambda pending_limit=10, resolved_limit=10, history_limit=10: {
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "summary": {
                "operational_status": "active",
                "high_risk": False,
                "pending_reviews": 1,
                "resolved_reviews": 4,
                "hold_history_entries": 2,
                "recommended_action": "Continue manual review of queued cases and monitor for new override patterns.",
            },
            "active_model": {
                "model_id": "oasis_current_baseline",
                "dataset": "oasis1",
                "run_name": "unit_run",
                "checkpoint_path": "checkpoint.pt",
                "promoted_at_utc": "2026-01-01T00:00:00+00:00",
                "decision_support_only": True,
                "clinical_disclaimer": "Decision-support only.",
                "default_threshold": 0.5,
                "recommended_threshold": 0.45,
                "threshold_calibration": {},
                "temperature_scaling": {"enabled": False, "temperature": 1.0},
                "confidence_policy": {"high_confidence_min": 0.85},
                "approval_status": "approved",
                "operational_status": "active",
                "benchmark": {},
                "promotion_decision": {"approved": True},
                "evidence": {},
                "validation_metrics": {},
                "test_metrics": {},
                "serving_restrictions": {"force_manual_review": False},
                "hold_decision": {},
                "review_monitoring": {"total_reviews": 5},
                "notes": [],
            },
            "pending_reviews": {"total": 1, "items": []},
            "resolved_reviews": {"total": 4, "items": []},
            "analytics": {
                "generated_at_utc": "2026-01-01T00:00:00+00:00",
                "scope": "active_model_id=oasis_current_baseline",
                "total_reviews": 5,
                "pending_reviews": 1,
                "resolved_reviews": 4,
                "adjudicated_reviews": 4,
                "overridden_reviews": 1,
                "confirmed_reviews": 3,
                "dismissed_reviews": 0,
                "override_rate": 0.25,
                "confirmation_rate": 0.75,
                "status_counts": {"confirmed": 3, "pending": 1, "overridden": 1},
                "action_counts": {"confirm_model_output": 3, "override_prediction": 1},
                "reviewer_counts": {"reviewer_a": 2},
                "confidence_level_counts": {"low": 5},
                "error_breakdown": {"false_positive": 1},
                "error_confidence_distribution": {"low": 1},
                "label_override_pairs": {"demented -> nondemented": 1},
                "model_breakdown": [],
                "average_error_probability_score": 0.53,
                "high_risk": False,
                "reviewer_agreement_available": False,
                "reviewer_agreement_note": "Single-review resolution workflow only.",
                "risk_signals": [],
                "notes": [],
            },
            "hold_history": {
                "total": 2,
                "items": [
                    {
                        "history_path": "hold_history.json",
                        "assessed_at_utc": "2026-01-01T00:00:00+00:00",
                        "policy_name": "oasis_operational_hold_v1",
                        "operational_status": "active",
                        "hold_applied": False,
                        "status_changed": False,
                        "trigger_codes": [],
                        "run_name": "unit_run",
                        "model_id": "oasis_current_baseline",
                        "summary": "No hold applied.",
                        "high_risk": False,
                        "total_reviews": 5,
                    }
                ],
            },
            "notes": ["Decision-support only."],
        },
    )
    client = TestClient(create_app())
    response = client.get("/reviews/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["operational_status"] == "active"
    assert payload["hold_history"]["total"] == 2
    assert payload["active_model"]["model_id"] == "oasis_current_baseline"


def test_resolve_review_endpoint_uses_service_layer(monkeypatch) -> None:
    """The review-resolution route should return the updated queued item."""

    monkeypatch.setattr(
        "src.api.routers.governance.resolve_review_queue_item_payload",
        lambda review_id, request: {
            "review_id": review_id,
            "status": "confirmed",
            "message": "resolved",
            "item": {
                "review_id": review_id,
                "inference_id": "inf-1",
                "trace_id": "trace-1",
                "scan_id": "scan-1",
                "subject_id": "OAS1_0001",
                "session_id": None,
                "source_path": "scan.nii.gz",
                "model_name": "densenet121_3d",
                "confidence_level": "low",
                "probability_score": 0.54,
                "output_path": "prediction.json",
                "status": "confirmed",
                "reason": "low_confidence_prediction",
                "resolution": {"reviewer_id": request.reviewer_id},
                "payload": {"review_flag": False},
                "created_at": "2026-01-01T00:00:00+00:00",
            },
        },
    )
    client = TestClient(create_app())
    response = client.post(
        "/reviews/review-1/resolve",
        json={
            "reviewer_id": "reviewer_a",
            "action": "confirm_model_output",
            "resolution_note": "Looks consistent with scan context.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "confirmed"
    assert payload["item"]["resolution"]["reviewer_id"] == "reviewer_a"


def test_predict_scan_route_uses_service_layer(monkeypatch) -> None:
    """The prediction route should delegate business logic to the service layer."""

    monkeypatch.setattr("src.api.routers.inference.build_scan_prediction_payload", lambda _request: _prediction_payload())
    client = TestClient(create_app())
    response = client.post(
        "/predict/scan",
        json={
            "scan_path": "scan.nii.gz",
            "checkpoint_path": "checkpoint.pt",
            "threshold": 0.5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_label"] == 1
    assert payload["decision_support_only"] is True
    assert "not a diagnosis" in payload["ai_summary"].lower()
    assert payload["confidence_level"] == "medium"


def test_explain_scan_route_uses_service_layer(monkeypatch) -> None:
    """The explainability route should return saved artifact metadata."""

    monkeypatch.setattr("src.api.routers.explainability.build_scan_explanation_payload", lambda _request: _explanation_payload())
    client = TestClient(create_app())
    response = client.post(
        "/explain/scan",
        json={
            "scan_path": "scan.nii.gz",
            "checkpoint_path": "checkpoint.pt",
            "slice_axis": "axial",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["method"] == "grad_cam_style_3d"
    assert payload["target_class_name"] == "demented"
    assert "diagnosis" in payload["limitations"][0].lower()
    assert payload["explanation_quality"] == "uncertain"


def test_longitudinal_report_route_uses_service_layer(monkeypatch) -> None:
    """The longitudinal report route should return timeline-ready report metadata."""

    monkeypatch.setattr(
        "src.api.routers.longitudinal.build_saved_longitudinal_report_payload",
        lambda _request: {
            "subject_id": "OAS1_9000",
            "report_type": "longitudinal_tracking",
            "timepoint_count": 1,
            "timepoints": [],
            "timeline": [],
            "interval_changes": [],
            "trend_summaries": [],
            "alerts": [],
            "feature_configs": [],
            "warnings": ["Only one timepoint is available."],
            "limitations": ["Decision-support only."],
            "output_path": "longitudinal_report.json",
        },
    )
    client = TestClient(create_app())
    response = client.post("/longitudinal/report", json={"subject_id": "OAS1_9000"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["subject_id"] == "OAS1_9000"
    assert payload["output_path"] == "longitudinal_report.json"


def test_longitudinal_report_route_builds_timeline_from_records(tmp_path, monkeypatch) -> None:
    """The API should support the general timeline/trend longitudinal engine."""

    from src.configs.runtime import AppSettings

    settings = AppSettings(
        project_root=tmp_path / "alz_backend",
        workspace_root=tmp_path,
        collection_root=tmp_path,
        data_root=tmp_path / "alz_backend" / "data",
        outputs_root=tmp_path / "alz_backend" / "outputs",
        kaggle_source_root=tmp_path,
        oasis_source_root=tmp_path / "OASIS",
    )
    settings.data_root.mkdir(parents=True)
    settings.outputs_root.mkdir(parents=True)
    monkeypatch.setattr("src.api.services.get_app_settings", lambda: settings)

    client = TestClient(create_app())
    response = client.post(
        "/longitudinal/report",
        json={
            "subject_id": "OAS1_9000",
            "output_name": "api_longitudinal_unit",
            "records": [
                {
                    "subject_id": "OAS1_9000",
                    "session_id": "OAS1_9000_MR1",
                    "visit_order": 1,
                    "scan_timestamp": "2001-01-01",
                    "source_path": "scan1.hdr",
                    "dataset": "oasis1",
                    "volumetric_features": {"left_hippocampus_volume_mm3": 3200.0},
                    "model_probabilities": {"ad_like_probability": 0.20},
                },
                {
                    "subject_id": "OAS1_9000",
                    "session_id": "OAS1_9000_MR2",
                    "visit_order": 2,
                    "scan_timestamp": "2002-01-01",
                    "source_path": "scan2.hdr",
                    "dataset": "oasis1",
                    "volumetric_features": {"left_hippocampus_volume_mm3": 3008.0},
                    "model_probabilities": {"ad_like_probability": 0.32},
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["report_type"] == "longitudinal_tracking"
    assert payload["timepoint_count"] == 2
    assert payload["timeline"][0]["session_id"] == "OAS1_9000_MR1"
    assert payload["trend_summaries"]
    assert payload["alerts"]


def test_api_key_auth_can_protect_ai_routes(monkeypatch) -> None:
    """Protected AI routes should require X-API-Key when configured."""

    monkeypatch.setenv("ALZ_API_KEY", "secret-dev-key")
    monkeypatch.setattr("src.api.routers.inference.build_scan_prediction_payload", lambda _request: _prediction_payload())
    client = TestClient(create_app())

    blocked = client.post(
        "/predict/scan",
        json={"scan_path": "scan.nii.gz", "checkpoint_path": "checkpoint.pt"},
    )
    allowed = client.post(
        "/predict/scan",
        headers={"X-API-Key": "secret-dev-key"},
        json={"scan_path": "scan.nii.gz", "checkpoint_path": "checkpoint.pt"},
    )

    assert blocked.status_code == 401
    assert allowed.status_code == 200
