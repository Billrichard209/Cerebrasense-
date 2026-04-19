# Scripts

Place reproducible entry-point scripts here for training, evaluation, preprocessing, and maintenance jobs.

Key OASIS productization entrypoints:

- `train_oasis_colab.py`: canonical Colab OASIS training and promotion wrapper
- `import_promoted_oasis_run.py`: import a promoted Colab OASIS run into local `outputs/`
- `check_oasis_productization.py`: verify local registry, evidence, and synced `backend_runtime` alignment
- `build_oasis_demo_bundle.py`: build one local demo bundle from the active registry and API surface
- `build_oasis_baseline_comparison.py`: compare the active local OASIS baseline against an imported candidate
- `build_oasis_presentation_summary.py`: turn comparison/productization artifacts into a presentation-ready status summary
- `build_scope_evidence_report.py`: regenerate the scope-aligned evidence report
