# OASIS Drive Upload Checklist

Use this when preparing the OASIS upload bundle for Google Drive.

This is the project's **primary 3D evidence branch**, so we keep this dataset separate from Kaggle.

## Recommended Approach

Do **not** upload the full raw OASIS archive if upload size is a problem.

Instead, build and upload the smaller labeled bundle produced by:

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\build_oasis_upload_bundle.py --materialize-mode hardlink
```

## What The Bundle Contains

- `OASIS/`
  Contains the labeled session subset the backend currently uses.
- `backend_reference/oasis1_manifest_relative.csv`
- `backend_reference/oasis1_manifest_summary.json`
- `backend_reference/oasis1_manifest_dropped_rows.csv`
- `backend_reference/oasis_session_index.csv`
- `backend_reference/oasis_upload_bundle_summary.json`
- `README.md`

## Recommended Google Drive Layout

```text
MyDrive/
  cerebrasense/
    data/
      oasis1/
        upload_bundle/
          OASIS/
          backend_reference/
          README.md
```

## What To Upload

Upload the whole generated bundle folder.

## What Not To Upload

Do not upload:

- the entire original OASIS archive if the bundle is enough for current work
- `.venv/`
- `outputs/`
- backend caches
- unrelated local files

## Later In Colab

Point the OASIS source path to the `OASIS/` directory inside the uploaded bundle.

Example shape:

```text
/content/drive/MyDrive/cerebrasense/data/oasis1/upload_bundle/OASIS
```

Then rebuild the manifest from that subset in the Colab environment if needed.

## Scope Reminder

- OASIS is the primary 3D structural MRI branch
- use this bundle for current OASIS-first training and evaluation
- do not merge OASIS with Kaggle implicitly
- OASIS-2 remains a future longitudinal extension
