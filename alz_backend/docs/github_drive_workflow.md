# GitHub And Drive Workflow

This project now follows a narrower and more honest scope:

- `OASIS-1` is the primary 3D structural MRI evidence track
- `Kaggle Alzheimer` is a separate 2D comparison branch
- `OASIS-2` is the next dataset target for real longitudinal strength

Because of that, code and data should not travel together as one giant folder.

## Recommended Split

- `GitHub`: backend code, configs, docs, notebooks, small reproducibility assets
- `Google Drive`: raw datasets, extracted dataset folders, checkpoints, outputs
- `Local laptop`: working copy for development, debugging, and smaller runs
- `Colab`: training and heavier evaluation

## What Goes To GitHub

Keep these:

- `alz_backend/src/`
- `alz_backend/scripts/`
- `alz_backend/configs/`
- `alz_backend/docs/`
- `alz_backend/notebooks/`
- `alz_backend/requirements.txt`
- `alz_backend/requirements-colab.txt`
- workspace `.gitignore`

Do not push these:

- `.venv/`
- `alz_backend/outputs/`
- raw OASIS data
- raw Kaggle Alzheimer data
- future OASIS-2 raw data
- large local zip backups

## What Goes To Google Drive

Recommended Drive layout:

```text
MyDrive/
  Cerebrasensecloud/
    OASIS-1/
    OASIS-2/
    kaggle_alz_upload_bundle/
    backend_runtime/
```

## Best Practical Workflow

1. Keep your main backend code in GitHub.
2. Keep raw datasets in Google Drive as extracted folders.
3. In Colab, mount Drive.
4. Clone or sync the backend code.
5. Point training configs to the Drive-mounted dataset folders.
6. Save outputs/checkpoints back to `backend_runtime/` on Drive.

## Upload Advice

- For upload convenience, you can zip datasets locally first.
- For actual training, extracted folders are better than zips.
- Keep OASIS and Kaggle fully separate on Drive.
- Do not upload the whole workspace with outputs and `.venv` mixed in.

## Scope Reminder

This backend is not a generic brain-scan platform.

The strongest current use is:

- structural MRI decision-support research
- OASIS-first 3D evidence
- future OASIS-2 longitudinal expansion

Kaggle remains secondary and should not replace the OASIS promotion path.
