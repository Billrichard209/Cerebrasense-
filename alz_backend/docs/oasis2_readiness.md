# OASIS-2 Readiness

OASIS-2 is the most natural future dataset extension for this backend because it
strengthens longitudinal subject tracking without changing the OASIS-first 3D
MRI direction.

This readiness workflow is intentionally modest:

- it checks whether a likely OASIS-2 source root exists
- it summarizes likely 3D image formats and metadata files
- it looks for OASIS-2-style subject/session identifiers such as `OAS2_0001`
  and `OAS2_0001_MR1`
- it estimates whether repeated-subject structure is visible yet

It does **not**:

- build an OASIS-2 manifest
- claim OASIS-2 is already integrated into training or evaluation
- silently merge OASIS-2 with OASIS-1

It now also understands split raw layouts such as:

- `OAS2_RAW_PART1/`
- `OAS2_RAW_PART2/`

## Default Source Resolution

The checker looks for OASIS-2 in this order:

1. `--source-root`
2. `ALZ_OASIS2_SOURCE_DIR`
3. auto-detected candidates such as:
   - `../OASIS2`
   - `../OASIS-2`
   - parent folders that contain `OAS2_RAW_PART1` and `OAS2_RAW_PART2`

## Run The Check

From `archive (1)`:

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\check_oasis2_readiness.py
```

Or point it to a specific dataset folder:

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\check_oasis2_readiness.py --source-root "C:\path\to\OASIS2"
```

## Build A Raw Inventory

If the dataset is present locally, you can now build a raw structural inventory:

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\build_oasis2_inventory.py
```

Or point it at the parent folder that contains the split raw parts:

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\build_oasis2_inventory.py --source-root "C:\path\to\Cerebrasense"
```

## Build An Unlabeled Session Manifest

With only the raw website download, the best next step is an unlabeled session manifest.
This selects one representative structural acquisition per session and produces
timeline-ready longitudinal records without inventing labels.

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\build_oasis2_session_manifest.py --source-root "C:\path\to\Cerebrasense"
```

## Build One Onboarding Bundle

If you want one single folder for review, onboarding, and planning, build the
local-first onboarding bundle:

```powershell
.\build_oasis2_onboarding_bundle.cmd --source-root "C:\path\to\Cerebrasense"
```

That bundle will:

- regenerate the readiness report
- rebuild the raw inventory
- rebuild the unlabeled session manifest
- package the key docs and artifacts into one review folder

## Check The OASIS-2 Adapter Stub

The next integration step after the onboarding bundle is the first dedicated
OASIS-2 manifest adapter stub:

```powershell
.\build_oasis2_adapter.cmd
```

That adapter check:

- validates the current unlabeled `oasis2_session_manifest.csv`
- confirms the adapter is limited to onboarding and longitudinal preparation
- writes a status report under `outputs/reports/onboarding/`
- fails the idea that OASIS-2 is already ready for supervised training

## When To Upload OASIS-2 To Google Drive

Do **not** upload OASIS-2 to Google Drive yet if we are only doing:

- local readiness checks
- raw inventory generation
- unlabeled session manifest building
- onboarding and planning work

Upload OASIS-2 to Drive only when one of these becomes true:

- we open a real **Colab** workflow for OASIS-2
- we need a remote runtime to preprocess or evaluate OASIS-2
- we have a dedicated OASIS-2 adapter/split workflow that actually needs cloud execution

For the current phase, the correct posture is:

- keep OASIS-2 **local**
- keep OASIS-1 as the active evidence/training path
- treat OASIS-2 as the next onboarding branch, not a Drive-backed training dataset

## Saved Outputs

Reports are saved under:

```text
alz_backend/outputs/reports/readiness/
```

Typical artifacts:

- `oasis2_readiness.json`
- `oasis2_readiness.md`
- `oasis2_raw_inventory.csv`
- `oasis2_raw_inventory_dropped_rows.csv`
- `oasis2_raw_inventory_summary.json`
- `oasis2_session_manifest.csv`
- `oasis2_longitudinal_records.csv`
- `oasis2_subject_summary.csv`
- `oasis2_session_manifest_summary.json`

## How To Read The Result

- `pass`
  The folder looks promising for a future OASIS-2 adapter.
- `warn`
  OASIS-2 is missing locally or some key signals are still incomplete.
- `fail`
  The supplied path is invalid for a dataset root.

Even a `pass` here only means the raw layout looks workable. A dedicated
manifest adapter, label mapping policy, and subject-safe split strategy still
need to be implemented before OASIS-2 becomes part of the evidence pipeline.

With raw-only OASIS-2, the backend can still support:

- preprocessing and transform previews
- structural measurement workflows
- subject/session indexing
- future longitudinal preparation

It should **not** yet be treated as a labeled classification dataset.
