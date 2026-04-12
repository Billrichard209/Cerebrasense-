# Backend Readiness Check

The readiness checker is a development and handoff aid. It verifies important
backend structure and artifacts without running expensive training or inference.
It is not a production compliance audit.

## Run

From `archive (1)`:

```powershell
cmd /c check_backend_readiness.cmd
```

From `alz_backend`:

```powershell
cmd /c check_backend_readiness.cmd
```

Use strict mode when you want warnings to fail CI-like checks:

```powershell
cmd /c check_backend_readiness.cmd --strict
```

## Outputs

The checker saves:

```text
outputs/reports/readiness/backend_readiness.json
outputs/reports/readiness/backend_readiness.md
```

## What It Checks

- project structure and required source modules
- OASIS/Kaggle configs
- generated manifests and reports when available
- optional trained checkpoints
- core Python package availability
- FastAPI app importability
- decision-support disclaimer wording

Warnings are allowed for optional artifacts, such as missing Kaggle manifest or
no trained checkpoint on a fresh setup. Failures indicate required backend
structure or dependency issues that should be fixed before demos or experiments.

