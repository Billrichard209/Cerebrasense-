# OASIS-2 Drive Upload Checklist

Use this file only when a real Google Drive or Colab workflow actually needs OASIS-2 outside the local machine.

This dataset is a **future longitudinal structural branch**, not a supervised classification branch.

If you are only doing:

- local readiness checks
- raw inventory generation
- unlabeled session manifest building
- onboarding and planning work

do **not** upload OASIS-2 to Drive yet. Keep it local-first until a remote runtime genuinely needs it.

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
- `backend_reference/oasis2_metadata_template.csv`
- `backend_reference/oasis2_metadata_template_summary.json`
- `backend_reference/oasis2_labeled_prep_manifest.csv`
- `backend_reference/oasis2_metadata_adapter_status.json`
- `backend_reference/oasis2_subject_safe_split_plan.csv`
- `backend_reference/oasis2_subject_safe_split_plan_summary.json`
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

If you use the current engineered path, the next manual step on your side is:

1. run `.\build_oasis2_upload_bundle.cmd`
2. wait for the bundle to finish under `alz_backend/outputs/exports/oasis2_upload_bundle/`
3. upload that whole folder to Drive
4. validate the finished bundle locally or after Drive sync with:

```powershell
.\check_oasis2_upload_bundle.cmd --bundle-root "C:\path\to\oasis2_upload_bundle"
```

That writes:

- `alz_backend/outputs/reports/onboarding/oasis2_upload_bundle_status.json`
- `alz_backend/outputs/reports/onboarding/oasis2_upload_bundle_status.md`

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

Then rebuild the raw inventory or unlabeled session manifest there if needed, using
the bundle root itself as `ALZ_OASIS2_SOURCE_DIR` or `--source-root`.

The uploaded bundle now also carries:

- the metadata template
- the metadata adapter status
- the labeled-prep manifest candidate
- the subject-safe split preview

So remote review can start from the same local onboarding state.

## Scope Reminder

- OASIS-2 is for longitudinal structural preparation
- OASIS-2 currently has no clinical labels in this workspace
- do not merge OASIS-2 into OASIS-1 supervised classification implicitly
- this bundle is for preprocessing, structural workflows, and timeline preparation
