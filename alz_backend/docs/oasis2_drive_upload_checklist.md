# OASIS-2 Drive Upload Checklist

Use this file when preparing the raw-only OASIS-2 longitudinal branch for Google Drive.

This dataset is a **future longitudinal structural branch**, not a supervised classification branch.

## Recommended Approach

Do not upload the full raw website download if upload size is a problem.

Build the portable bundle instead:

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\build_oasis2_upload_bundle.py --source-root "C:\Users\Nguyen Quang Minh\OneDrive\Desktop\Cerebrasense" --materialize-mode hardlink
```

That produces:

```text
alz_backend/outputs/exports/oasis2_upload_bundle/
```

## What The Bundle Contains

- selected files from `OAS2_RAW_PART1/` and `OAS2_RAW_PART2/`
- `backend_reference/oasis2_session_manifest_relative.csv`
- `backend_reference/oasis2_longitudinal_records_relative.csv`
- `backend_reference/oasis2_subject_summary.csv`
- `backend_reference/oasis2_raw_inventory.csv`
- `backend_reference/oasis2_raw_inventory_summary.json`
- `backend_reference/oasis2_session_manifest_summary.json`
- `backend_reference/oasis2_session_index.csv`
- `backend_reference/oasis2_upload_bundle_summary.json`
- `backend_reference/oasis2_readiness.json`
- `README.md`

## Recommended Google Drive Layout

```text
MyDrive/
  cerebrasense/
    data/
      oasis2/
        upload_bundle/
          OAS2_RAW_PART1/
          OAS2_RAW_PART2/
          backend_reference/
          README.md
```

## What To Upload

Upload the whole generated bundle folder.

## What Not To Upload

Do not upload:

- unrelated local files
- `.venv/`
- backend `outputs/` outside the generated bundle
- the full raw OASIS-2 download if the bundle is enough for current work

## Later In Colab

Point the OASIS-2 source path at the bundle root.

Example:

```text
/content/drive/MyDrive/cerebrasense/data/oasis2/upload_bundle
```

Then rebuild the raw inventory or unlabeled session manifest there if needed.

## Scope Reminder

- OASIS-2 is for longitudinal structural preparation
- OASIS-2 currently has no clinical labels in this workspace
- do not merge OASIS-2 into OASIS-1 supervised classification implicitly
- this bundle is for preprocessing, structural workflows, and timeline preparation
