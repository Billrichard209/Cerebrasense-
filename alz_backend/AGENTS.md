# AGENTS.md

## Mission

Build the backend core for an OASIS-first structural brain MRI decision-support platform with clear dataset boundaries, reproducible research practices, and production-like Python structure.

The working scientific scope is:

- structural MRI analysis for dementia-related classification research
- longitudinal scan tracking and structural trend reporting
- explainable and auditable decision-support workflows

The current scope is not a broad "brain scan AI" system and not a stand-alone Alzheimer diagnosis product.

## Operating Rules

1. OASIS-1 is the primary dataset for the first implementation pass.
2. Kaggle dataset code must stay separate unless explicit merge instructions are given.
3. Future OASIS-2 support is a natural longitudinal extension and should fit the same OASIS-first architecture.
4. Never perform silent label remapping. Any mapping must be explicit, documented, reviewed, and test-covered.
5. This codebase supports decision assistance only. Do not describe outputs as medical diagnosis.
6. Preserve provenance for data processing, checkpoints, metrics, and predictions.
7. Prefer small, typed, modular Python components over notebook-only logic.
8. Keep PHI/privacy risk in mind even when using public datasets; avoid normalizing unsafe habits into the architecture.
9. Put dataset-specific assumptions in dataset-specific modules, not in shared utilities.
10. Add or update tests when changing label logic, cohort filtering, or evaluation behavior.
11. Future patient management and research analytics should be added as new modules, not by overloading MRI core packages.
12. Do not broaden product claims beyond what OASIS-1, Kaggle, and future OASIS-2 can reasonably support.
13. Do not market the current system as a generic detector for arbitrary brain disorders.

## Coding Conventions

- Python 3.10+
- Use docstrings on public modules and functions
- Favor `pathlib.Path`, `dataclasses`, and explicit return types
- Keep shared helpers in `src/utils`
- Keep API concerns in `src/api`
- Keep volumetric and longitudinal logic outside generic model code

## Safety and Ethics Guardrails

- No claims of autonomous diagnosis
- No hidden dataset harmonization
- No hidden threshold tuning in inference scripts
- No patient-identifying data in logs, reports, or example payloads

## Directory Ownership

- `src/data`: dataset contracts and dataset-specific loaders
- `src/transforms`: preprocessing and normalization steps
- `src/models`: model definitions and metadata
- `src/training`: orchestration for training runs
- `src/evaluation`: metrics and validation routines
- `src/inference`: prediction entry points
- `src/explainability`: interpretable output artifacts
- `src/volumetrics`: structural and regional volume utilities
- `src/longitudinal`: visit-to-visit tracking logic
- `src/security`: governance, privacy, and safety helpers
