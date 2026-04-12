# Backend Session Rollout

This project is moving forward in deliberate sessions so the backend stays reliable while the AI stack gets stronger.

## Session 1
- Reviewer dashboard API
- Pending and resolved review views
- Review analytics endpoint
- Active-model hold history
- Compact reviewer-operations dashboard

Status:
- Implemented

## Session 2
- Experiment promotion workflow surface
- Candidate-vs-active comparison payloads
- Promotion-readiness summary

Status:
- Implemented

## Session 3
- Repeated subject-safe validation workflow
- Repeated split or cross-validation summaries
- Comparison reports for model stability

Status:
- Implemented

## Session 4
- External evaluation preparation
- Explicit external-cohort manifest and evaluation path
- Separate reporting for non-OASIS evidence

Status:
- Implemented

## Session 5
- Reviewer outcome learning loop
- Threshold tuning support from reviewer feedback
- Failure-pattern reports for future retraining decisions

Status:
- Implemented

Notes:
- OASIS and Kaggle stay separate unless explicitly configured otherwise.
- The current dataset-backed scope is structural MRI decision support for dementia-related research, not generic brain-disease detection.
- OASIS-2 is the most natural future dataset extension because it strengthens longitudinal evidence without changing the core OASIS-first architecture.
- OASIS-2 readiness reporting is now available as a separate pre-integration step before any future manifest or split builder is added.
- Backend outputs remain decision-support only and must not be treated as diagnosis.
- Each session should ship with tests and traceable saved artifacts where relevant.
