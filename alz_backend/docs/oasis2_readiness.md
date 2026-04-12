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

## Default Source Resolution

The checker looks for OASIS-2 in this order:

1. `--source-root`
2. `ALZ_OASIS2_SOURCE_DIR`
3. auto-detected candidates such as:
   - `../OASIS2`
   - `../OASIS-2`

## Run The Check

From `archive (1)`:

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\check_oasis2_readiness.py
```

Or point it to a specific dataset folder:

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\check_oasis2_readiness.py --source-root "C:\path\to\OASIS2"
```

## Saved Outputs

Reports are saved under:

```text
alz_backend/outputs/reports/readiness/
```

Typical artifacts:

- `oasis2_readiness.json`
- `oasis2_readiness.md`

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
