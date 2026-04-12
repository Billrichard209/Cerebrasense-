# External Validation Quickstart

Use this path only for a true outside cohort such as ADNI, AIBL, or another separately curated 3D MRI dataset.

This path is optional future work. The current backend can still move forward as
an OASIS-first structural MRI research system even when external cohorts are
not available yet.

## Rules

- Do not reuse OASIS through the external-cohort path.
- Do not reuse Kaggle through the external-cohort path.
- Do not auto-harmonize labels inside the evaluator.
- The manifest must already encode the binary convention used by the current OASIS model:
  - `0 = control / nondemented`
  - `1 = AD-like / demented`

## Required manifest fields

- `image`
- `label`
- `dataset`
- `dataset_type`

Recommended additional fields:

- `label_name`
- `subject_id`
- `session_id`
- `scan_timestamp`
- `meta`

Use the template at `configs/external_cohort_manifest_template.csv`.

## Build a manifest from files plus metadata

If your external cohort already has a metadata CSV, use the builder instead of hand-editing paths:

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\build_external_manifest.py `
  --images-root path\to\ADNI_scans `
  --dataset-name adni_pilot `
  --output-path alz_backend\data\interim\adni_pilot_manifest.csv `
  --metadata-csv-path path\to\ADNI_metadata.csv `
  --image-column image_path `
  --label-column label `
  --label-name-column diagnosis_name `
  --subject-id-column subject_id `
  --session-id-column session_id `
  --scan-timestamp-column scan_date `
  --meta-columns site scanner `
  --require-labels `
  --validate-for-evaluation
```

This writes both:

- the manifest CSV
- a `.build_report.json` file next to the manifest

## Dataset type

For the current OASIS MONAI 3D classifier, the manifest must use:

- `dataset_type = 3d_volumes`

If the outside dataset is slice-based or 2D, do not force it through this evaluation path.

## Example command

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\evaluate_external_cohort.py `
  --manifest-path path\to\external_manifest.csv `
  --checkpoint-path alz_backend\outputs\runs\oasis\oasis_baseline_rtx2050_gpu_seed42_split42\checkpoints\best_model.pt `
  --output-name adni_pilot_eval `
  --device cuda
```

## Expected outputs

- `metrics.json`
- `predictions.csv`
- `confusion_matrix.json/.csv/.png`
- `roc_curve.csv/.png`
- `summary_report.md`

## Honest limitation

The backend is ready for external evaluation, but external evidence does not exist until a true outside cohort manifest is provided and evaluated.
