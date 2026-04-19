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

## Build The Metadata Template

Once the unlabeled session manifest exists, generate the metadata mapping template:

```powershell
.\build_oasis2_metadata_template.cmd
```

This creates a CSV you can fill with explicit visit/clinical metadata such as:

- `age_at_visit`
- `clinical_status`
- `cdr_global`
- `mmse`
- `diagnosis_label`
- `diagnosis_label_name`
- `metadata_source`
- `split_group_hint`

## Check The Metadata Adapter

After filling the metadata template, validate the merge path with:

```powershell
.\build_oasis2_metadata_adapter.cmd
```

That step:

- merges the metadata template back onto the OASIS-2 session manifest
- writes a labeled-prep manifest candidate
- reports how many rows actually have candidate labels
- tells you honestly whether OASIS-2 is ready for labeled-manifest work

## Build The Subject-Safe Split Preview

After the metadata template exists, generate the planning-only split preview:

```powershell
.\build_oasis2_split_policy.cmd
```

This step:

- groups rows by `split_group_hint` or `subject_id`
- assigns deterministic subject-safe buckets
- marks conservative `holdout_candidate` and `development_candidate` roles
- avoids pretending that true train/val/test splits already exist

## Check Training Readiness

After the metadata adapter and split preview both exist, run the supervised
training gate:

```powershell
.\check_oasis2_training_readiness.cmd
```

This step:

- checks that every OASIS-2 session has both `label` and `label_name`
- enforces the current binary label policy `{0, 1}`
- verifies that the split plan covers every labeled `split_group_hint`
- attempts to materialize real subject-safe `train/val/test` manifests
- writes `oasis2_training_readiness.json` and `.md` under `outputs/reports/onboarding/`

If the report says `fail`, do not start training yet. Fix the metadata/template
coverage first and rerun the gate.

## Build Supervised Split Manifests

When the readiness gate passes, materialize the supervised OASIS-2 manifests:

```powershell
.\build_oasis2_supervised_splits.cmd
```

This step writes one report folder under `outputs/reports/` that contains:

- `oasis2_supervised_group_assignments.csv`
- `oasis2_train_manifest.csv`
- `oasis2_val_manifest.csv`
- `oasis2_test_manifest.csv`
- `oasis2_supervised_split_summary.json`
- `oasis2_supervised_split_summary.md`

## Train OASIS-2

Once the readiness gate passes, start the local research training runner with:

```powershell
.\train_oasis2.cmd
```

Or start with a tiny pipeline check:

```powershell
.\train_oasis2.cmd --dry-run
```

The training runner re-checks supervised readiness before it touches the model,
so it fails fast and points you at the saved readiness report if labels or split
coverage are still incomplete.

## Use The Uploaded Drive Bundle

If the uploaded Drive bundle now exists under:

```text
/content/drive/MyDrive/Cerebrasensecloud/OASIS-2
```

use the remote bundle gate instead of stitching the remote steps together by
hand. The canonical Colab entrypoints are:

- [oasis2_train.ipynb](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/oasis2_train.ipynb)
- `scripts/train_oasis2_colab.py`

That remote runner:

- validates the extracted upload bundle
- copies `backend_reference/oasis2_metadata_template.csv` into the runtime
- rebuilds the OASIS-2 manifest, metadata adapter output, and split plan from the bundle
- checks supervised readiness honestly
- starts training only when the label and split gates both pass

If labels are still missing, the run stops with a saved blocked summary instead
of pretending that training has started.

## When To Upload OASIS-2 To Google Drive

Do **not** upload OASIS-2 to Google Drive just for local-only onboarding if we
are only doing:

- local readiness checks
- raw inventory generation
- unlabeled session manifest building
- onboarding and planning work

Upload OASIS-2 to Drive only when one of these becomes true:

- we open a real **Colab** workflow for OASIS-2
- we need a remote runtime to preprocess or evaluate OASIS-2
- we have a dedicated OASIS-2 adapter/split workflow that actually needs cloud execution

That condition is now met at the bundle level. The repo can generate a portable
OASIS-2 upload bundle, and the remote bundle gate now exists too. When you
upload, upload the generated bundle folder rather than the whole raw workspace.

For the current phase, the correct posture is:

- keep OASIS-1 as the active evidence/training path
- use the uploaded OASIS-2 bundle for remote preparation and gated readiness checks
- do not expect supervised OASIS-2 training to start until `diagnosis_label` and `diagnosis_label_name` are filled for every session

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
- `oasis2_training_readiness.json`
- `oasis2_training_readiness.md`

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
