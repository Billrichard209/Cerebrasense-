# Scripts

Place reproducible entry-point scripts here for training, evaluation, preprocessing, and maintenance jobs.

Key OASIS productization entrypoints:

- `train_oasis_colab.py`: canonical Colab OASIS training and promotion wrapper
- `train_oasis2.py`: guarded supervised OASIS-2 training runner once labels and splits are ready
- `train_oasis2_colab.py`: uploaded-bundle OASIS-2 Colab gate that rebuilds runtime manifests and only trains when labels are ready
- `import_promoted_oasis_run.py`: import a promoted Colab OASIS run into local `outputs/`
- `check_oasis_productization.py`: verify local registry, evidence, and synced `backend_runtime` alignment
- `build_oasis_demo_bundle.py`: build one local demo bundle from the active registry and API surface
- `build_oasis_baseline_comparison.py`: compare the active local OASIS baseline against an imported candidate
- `build_oasis_presentation_summary.py`: turn comparison/productization artifacts into a presentation-ready status summary
- `build_project_status_bundle.py`: collect the current backbone docs and latest status artifacts into one review bundle
- `build_scope_evidence_report.py`: regenerate the scope-aligned evidence report
- `check_oasis2_training_readiness.py`: verify whether OASIS-2 labels and split planning are actually sufficient for supervised training
- `build_oasis2_supervised_splits.py`: materialize subject-safe OASIS-2 train/val/test manifests from the labeled-prep manifest
