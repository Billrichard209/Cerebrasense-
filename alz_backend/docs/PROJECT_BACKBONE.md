# Project Backbone

This is the fastest single-file entrypoint for understanding what this project is, where it stands, and what to read next.

## What This Project Is

The current project is best understood as:

- an **OASIS-first structural brain MRI research backend**
- for **dementia-related decision support and scan analysis**
- with **explainability, structural summaries, longitudinal support, and governance workflows**
- delivered as **research/decision-support software**, not diagnosis software

This project is intentionally narrower than a general-purpose brain imaging AI system.

## North Star

The main goal right now is:

- keep one honest, reproducible OASIS-first baseline working across **Colab**, **Google Drive**, and the **local backend**
- support demo-ready local inference, explanation, review, and longitudinal workflows
- keep **Kaggle** as a separate secondary benchmark branch
- prepare **OASIS-2** as the next major longitudinal expansion only after the OASIS baseline is stable

## Current Position

Right now the project has:

- a working local OASIS baseline
- a working Colab-to-Drive OASIS training flow
- a local compare-first workflow for imported candidate baselines
- demo bundles and presentation-ready summary artifacts

The current practical recommendation is:

- keep the stronger current local OASIS baseline active
- keep the imported Colab `v3` run as a validated candidate
- do not switch the local serving default until a future candidate clearly wins on held-out evidence without increasing review burden

## Read These First

If you want the shortest high-value reading order, use this:

1. [project_scope.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/project_scope.md)
   This defines what the backend is, what datasets mean, and what is out of scope.
2. [oasis_productization_workflow.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/oasis_productization_workflow.md)
   This is the canonical execution path for the current OASIS phase.
3. [oasis_presentation_summary.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/outputs/reports/presentation/oasis_presentation_summary.md)
   This is the current project-status briefing.

## Full Reading Map

### Scope and identity

- [project_scope.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/project_scope.md)
- [README.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/README.md)
- [AGENTS.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/AGENTS.md)

### Canonical workflow

- [oasis_productization_workflow.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/oasis_productization_workflow.md)
- [github_drive_workflow.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/github_drive_workflow.md)
- [colab_cerebrasensecloud_quickstart.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/colab_cerebrasensecloud_quickstart.md)

### Evidence and decision artifacts

- [scope_aligned_evidence_report.json](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/outputs/reports/evidence/scope_aligned_evidence_report.json)
- [oasis_productization_status.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/outputs/reports/productization/oasis_productization_status.md)
- [oasis_baseline_comparison.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/outputs/reports/comparison/oasis_baseline_comparison.md)
- [oasis_presentation_summary.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/outputs/reports/presentation/oasis_presentation_summary.md)
- [project_status_bundle.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/outputs/reports/status/current_project_status/project_status_bundle.md)

### Next expansion

- [oasis2_readiness.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/oasis2_readiness.md)
- [oasis2_onboarding_bundle.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/outputs/reports/onboarding/current_oasis2_onboarding/oasis2_onboarding_bundle.md)

## Operational Backbone

These are the main command/launcher entrypoints for the current phase:

- [import_promoted_oasis_run.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/import_promoted_oasis_run.cmd)
- [check_oasis_productization.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/check_oasis_productization.cmd)
- [build_oasis_demo_bundle.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/build_oasis_demo_bundle.cmd)
- [build_oasis_baseline_comparison.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/build_oasis_baseline_comparison.cmd)
- [build_oasis_presentation_summary.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/build_oasis_presentation_summary.cmd)
- [build_project_status_bundle.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/build_project_status_bundle.cmd)
- [build_oasis2_onboarding_bundle.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/build_oasis2_onboarding_bundle.cmd)
- [build_oasis2_adapter.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/build_oasis2_adapter.cmd)
- [build_oasis2_metadata_template.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/build_oasis2_metadata_template.cmd)
- [build_oasis2_metadata_adapter.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/build_oasis2_metadata_adapter.cmd)
- [build_oasis2_split_policy.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/build_oasis2_split_policy.cmd)
- [check_oasis2_training_readiness.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/check_oasis2_training_readiness.cmd)
- [build_oasis2_supervised_splits.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/build_oasis2_supervised_splits.cmd)
- [train_oasis2.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/train_oasis2.cmd)
- [build_oasis2_upload_bundle.cmd](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/build_oasis2_upload_bundle.cmd)

## Best Next Steps

The best next moves for the project are:

1. Keep the current local OASIS baseline active and use the new comparison/presentation artifacts for demos and status reviews.
2. Refresh the Kaggle branch once as a secondary benchmark, but do not let it replace the OASIS promotion path.
3. Use repeated-split refreshes or targeted error analysis if you do more OASIS research before adding new datasets.
4. When OASIS-2 labels are ready, run the new readiness gate and supervised split builder before attempting any OASIS-2 training run.
5. Use the uploaded OASIS-2 bundle with the new remote bundle gate, but keep OASIS-2 training blocked until the metadata template is filled and the readiness gate passes.

## Simple Rule

If you are unsure where to start:

- read [project_scope.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/project_scope.md)
- then read [oasis_productization_workflow.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/oasis_productization_workflow.md)
- then read [oasis_presentation_summary.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/outputs/reports/presentation/oasis_presentation_summary.md)

That trio gives you the project identity, the execution path, and the current decision state.
