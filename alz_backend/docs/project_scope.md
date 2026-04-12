# Project Scope

This document defines the current evidence-backed scope of the backend.

## Working Scope

The backend is currently best framed as:

- an **OASIS-first structural brain MRI research backend**
- for **dementia-related classification research**
- with **explainable inference, structural summaries, and longitudinal tracking**
- delivered as **decision-support software**, not diagnosis software

This is intentionally narrower than a general-purpose "brain scan AI" system.

## Why This Scope Fits The Data

### OASIS-1

OASIS-1 is the strongest dataset currently available in this workspace for:

- 3D structural MRI pipelines
- OASIS-first MONAI training and evaluation
- governance, calibration, and review workflows
- structural proxy and visit-aware reporting foundations

### Kaggle Alzheimer Dataset

The current Kaggle dataset is useful, but only as a separate branch:

- it is primarily a **2D slice/image dataset**
- it does not provide the same subject-level 3D evidence as OASIS
- it is useful for secondary comparison experiments and engineering validation
- it must not be silently treated as equivalent to OASIS labels or clinical cohorts

### OASIS-2

OASIS-2 is the most natural future expansion because it would strengthen:

- longitudinal scan history
- repeated-visit comparison
- subject-level progression summaries
- longitudinal model evaluation

See [oasis2_readiness.md](/c:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/oasis2_readiness.md) for the current onboarding/readiness workflow.

## In Scope

- OASIS-first 3D MRI classification research
- Kaggle as a separate 2D comparison branch
- structural volumetric proxy reporting
- explainability for current model outputs
- longitudinal trend summaries
- governance, review queue, calibration, and promotion workflows
- backend-ready schemas for later patient management and research analytics

## Out Of Scope For Now

The current datasets do not support these claims responsibly:

- generic detection of arbitrary brain diseases
- tumor, stroke, hemorrhage, trauma, or seizure detection
- autonomous Alzheimer diagnosis
- clinical deployment claims
- silent OASIS/Kaggle harmonization
- "all brain MRI" marketing language

## Official Problem Statement

The current backend supports research on structural brain MRI decision support for dementia-related pattern analysis, explainable scan review, and longitudinal tracking, using OASIS as the primary 3D reference dataset and Kaggle as a separate secondary experiment branch.

## Naming Note

The repository and package names still include `alz` because of project history. That does not change the narrower scientific scope described here.
