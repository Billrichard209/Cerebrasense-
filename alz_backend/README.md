# Structural MRI Backend Core (`alz_backend`)

Research-grade Python backend scaffold for an OASIS-first structural brain MRI decision-support platform focused on dementia-related classification research, structural volumetrics, longitudinal tracking, and secure backend workflows.

The repository name stays `alz_backend`, but the current evidence-backed scope is narrower than a general-purpose "Alzheimer diagnosis system" or a broad "brain scan AI" product.

## Scope

- Primary dataset: **OASIS-1**
- Secondary dataset: **Kaggle Alzheimer dataset**, kept fully separate as a separate 2D research branch
- Planned future dataset: **OASIS-2** for stronger longitudinal subject tracking
- Product posture: **decision-support software**, not diagnostic software
- Backend-first architecture with room for future patient management, nurse workflows, and research analytics

## Current Problem Statement

The current backend is best framed as:

- an **OASIS-first structural MRI research backend**
- for **dementia-related risk stratification and scan comparison**
- with **longitudinal tracking and structural summary support**
- while keeping **Kaggle evidence separate** from the OASIS 3D branch

This is a stronger and more honest fit than claiming broad brain-disease coverage or stand-alone Alzheimer diagnosis.

For the fuller scope statement, see [docs/project_scope.md](/c:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/project_scope.md).
For future OASIS-2 intake planning, see [docs/oasis2_readiness.md](/c:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/oasis2_readiness.md).

## Core Principles

- Do not silently merge OASIS and Kaggle data pipelines
- Do not silently remap labels or class names
- Keep preprocessing, metadata handling, and evaluation logic dataset-specific
- Preserve auditability for model outputs and longitudinal summaries
- Treat patient-facing use cases as safety-sensitive
- Keep product claims aligned with the datasets actually available in this repository

## Project Layout

```text
alz_backend/
  configs/                # Runtime and deployment configuration notes/templates
  data/                   # Raw, interim, processed, metadata, and sample inputs
  outputs/                # Checkpoints, metrics, reports, predictions, logs
  scripts/                # Operational scripts for training/evaluation jobs
  src/                    # Main application code
  tests/                  # Smoke and regression tests
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the starter API:

```bash
uvicorn src.api.main:app --reload
```

## Local Colab Handoff

After a Colab OASIS run is trained and exported to Google Drive, the clean local handoff is:

1. Import the promoted run into local `outputs/`
2. Use the imported registry-backed checkpoint for local scan inference

From the workspace root you can run:

```powershell
.\import_promoted_oasis_run.cmd `
  --source-run-root "C:\path\to\Drive\Cerebrasensecloud\training_runs\oasis\oasis_colab_full_v3_auroc_monitor" `
  --source-registry-path "C:\path\to\Drive\Cerebrasensecloud\model_registry\oasis_current_baseline.json" `
  --overwrite
```

Then run local inference through the active imported registry:

```powershell
.\predict_scan.cmd `
  --scan-path "C:\path\to\scan.hdr" `
  --device cpu `
  --use-registry-threshold
```

Useful entrypoints:

- [import_promoted_oasis_run.py](/c:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/scripts/import_promoted_oasis_run.py)
- [predict_scan.py](/c:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/scripts/predict_scan.py)
- [import_promoted_oasis_run.cmd](/c:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/import_promoted_oasis_run.cmd)
- [predict_scan.cmd](/c:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/predict_scan.cmd)

## Dataset Policy

### OASIS-1

OASIS-1 is the primary reference dataset for the current backend implementation. It is the main source of evidence for:

- 3D structural MRI classification research
- explainable inference on structural MRI volumes
- governance and review workflows
- structural proxy reporting

New interfaces should support OASIS first unless there is a clear reason to generalize immediately.

### OASIS-2

OASIS-2 is the most natural future expansion of the current scope. It fits the existing architecture well because it can strengthen:

- subject-level scan history
- longitudinal progression summaries
- repeated-scan structural trend analysis
- future split logic for visit-aware evaluation

### Kaggle

Kaggle support is intentionally separate. In the current repository it is best treated as:

- a separate **2D slice-based research branch**
- useful for engineering experiments and comparison studies
- not equivalent to the OASIS 3D MRI branch

Keep label definitions, preprocessing assumptions, metadata handling, and evaluation outputs isolated from OASIS unless an explicit cross-dataset study is planned and documented.

## Out Of Scope For Now

The current datasets do **not** justify these claims:

- broad "any brain scan" detection
- tumor, stroke, hemorrhage, trauma, or seizure detection
- autonomous Alzheimer diagnosis
- clinical-grade deployment claims
- silent harmonization across OASIS and Kaggle

If the scope broadens later, it should happen only after new datasets and new validation evidence are added.

## Safety Note

This repository is for structural brain MRI decision-support research and engineering workflows. It must not be presented as a stand-alone diagnostic system.
