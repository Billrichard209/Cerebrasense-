# OASIS Productization Workflow

This is the canonical path for taking the chosen OASIS Colab baseline and turning it into a stable local backend/demo baseline.

## Goal

Keep one honest OASIS-first baseline aligned across:

- Colab training
- Google Drive `backend_runtime`
- local imported registry/checkpoint
- evidence reporting
- local API/demo flows

## Canonical Flow

1. Train and promote in Colab with [train.ipynb](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/train.ipynb)
2. Persist the promoted run under `Cerebrasensecloud/backend_runtime`
3. Import that promoted run locally
4. Check local/evidence/runtime alignment
5. Build one local demo bundle from the active registry

## Colab Source Of Truth

The cloud source of truth is:

```text
Cerebrasensecloud/backend_runtime/
```

Important contents:

```text
backend_runtime/
  data/
  outputs/
    runs/oasis/<run_name>/
    model_registry/oasis_current_baseline.json
```

Use `train.ipynb` as the canonical OASIS Colab entrypoint. Do not create alternate notebook flows for the same baseline.

## Local Import

From the workspace root:

```powershell
.\import_promoted_oasis_run.cmd `
  --source-runtime-root "C:\path\to\Drive\Cerebrasensecloud\backend_runtime" `
  --overwrite
```

This imports:

- the promoted run into local `outputs/runs/oasis/...`
- the promoted registry into local `outputs/model_registry/oasis_current_baseline.json`

## Alignment Check

After import, check that local, evidence, and the synced runtime all agree:

```powershell
.\check_oasis_productization.cmd `
  --source-runtime-root "C:\path\to\Drive\Cerebrasensecloud\backend_runtime" `
  --expected-run-name "oasis_colab_full_v3_auroc_monitor"
```

This checks:

- local active registry exists
- local checkpoint/config paths resolve
- local active run matches the expected canonical run
- `scope_aligned_evidence_report.json` matches the active registry
- repeated-split OASIS support is still visible
- synced `backend_runtime` matches the local active baseline

## Demo Bundle

Build one reproducible local demo bundle from the active registry and API surface:

```powershell
.\build_oasis_demo_bundle.cmd `
  --scan-path "C:\path\to\scan.hdr" `
  --device cpu
```

Saved bundle contents include:

- root/health payloads
- active model metadata
- review dashboard payload
- one prediction example
- one explanation example
- one longitudinal example
- copied active registry
- latest readiness report
- latest scope evidence report

Default output root:

```text
alz_backend/outputs/reports/demo/<output_name>/
```

## Manual Decisions

The human review points are:

- confirm the Colab `repo_commit` is the latest pushed revision
- confirm `backend_runtime` contains the promoted run and registry
- confirm the imported local baseline should become the working local default
- review demo outputs for clarity and portfolio/demo readiness

## Ground Rules

- OASIS remains the primary 3D evidence track
- Kaggle remains a separate 2D comparison branch
- OASIS-2 stays in readiness/preparation until the productized OASIS baseline is stable
- do not run new random OASIS sweeps until cloud and local are aligned again
