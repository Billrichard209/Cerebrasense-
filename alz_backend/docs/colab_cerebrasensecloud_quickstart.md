# Colab Quickstart for `Cerebrasensecloud`

Use this when your Colab datasets live in Google Drive under:

```text
/content/drive/MyDrive/Cerebrasensecloud/
```

This project now has two hardened Colab entrypoints:

- OASIS primary branch: [train.ipynb](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/train.ipynb)
- Kaggle secondary branch: [kaggle_train.ipynb](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/kaggle_train.ipynb)

Both notebooks follow the same pattern:

- clone code into `/content/Cerebrasense-` for speed
- stage dataset files from Drive into local `/content` storage when requested
- persist `data/` and `outputs/` under a Drive runtime root so manifests, checkpoints, metrics, and summaries survive Colab restarts

## Recommended Drive Layout

```text
/content/drive/MyDrive/Cerebrasensecloud/
  OASIS-1/
  OASIS-2/
  kaggle_alz_upload_bundle/
  backend_runtime/
```

Notes:

- `OASIS-1/` should contain the OASIS upload bundle data used by the OASIS notebook.
- `kaggle_alz_upload_bundle/` should be the whole Kaggle bundle folder produced by `build_kaggle_upload_bundle.py`.
- `backend_runtime/` will be created automatically if missing.

## Common Colab Setup

Use these checks before running either notebook.

### Cell 1: Mount Drive and Verify Roots

```python
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/Cerebrasensecloud")
OASIS1 = DRIVE_ROOT / "OASIS-1"
OASIS2 = DRIVE_ROOT / "OASIS-2"
KAGGLE_BUNDLE = DRIVE_ROOT / "kaggle_alz_upload_bundle"
RUNTIME_ROOT = DRIVE_ROOT / "backend_runtime"

for name, path in {
    "DRIVE_ROOT": DRIVE_ROOT,
    "OASIS1": OASIS1,
    "OASIS2": OASIS2,
    "KAGGLE_BUNDLE": KAGGLE_BUNDLE,
    "RUNTIME_ROOT": RUNTIME_ROOT,
}.items():
    print(name, path.exists(), path)
```

### Cell 2: Clone the Repo and Install Dependencies

```python
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/content/Cerebrasense-")
BACKEND_ROOT = REPO_ROOT / "alz_backend"

if REPO_ROOT.exists():
    shutil.rmtree(REPO_ROOT)

subprocess.run(
    ["git", "clone", "https://github.com/Billrichard209/Cerebrasense-.git"],
    cwd="/content",
    check=True,
)

if not BACKEND_ROOT.exists():
    raise FileNotFoundError(f"Expected backend after clone: {BACKEND_ROOT}")

requirements = BACKEND_ROOT / "requirements-colab.txt"
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements)], check=True)

print("repo_root =", REPO_ROOT)
print("backend_root =", BACKEND_ROOT)
```

## Recommended OASIS Flow

Use [train.ipynb](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/train.ipynb).

That notebook already handles:

- Drive mount
- fresh clone
- dependency install
- persistent runtime roots under `backend_runtime/`
- local staging for OASIS
- manifest rebuild and validation
- safe resume or reuse of the same `RUN_NAME`
- threshold calibration and promotion for the OASIS baseline

After the promoted run is synced back to Windows, the canonical local handoff is:

```powershell
.\import_promoted_oasis_run.cmd `
  --source-runtime-root "C:\path\to\Drive\Cerebrasensecloud\backend_runtime" `
  --overwrite
```

Then verify alignment:

```powershell
.\check_oasis_productization.cmd `
  --source-runtime-root "C:\path\to\Drive\Cerebrasensecloud\backend_runtime" `
  --expected-run-name "oasis_colab_full_v3_auroc_monitor"
```

Then build a local demo bundle:

```powershell
.\build_oasis_demo_bundle.cmd `
  --scan-path "C:\path\to\scan.hdr" `
  --device cpu
```

If you prefer the script directly, the main entrypoint is:

```bash
cd /content/Cerebrasense-/alz_backend
python scripts/train_oasis_colab.py \
  --project-root /content/Cerebrasense-/alz_backend \
  --runtime-root /content/drive/MyDrive/Cerebrasensecloud/backend_runtime \
  --oasis-source-dir /content/drive/MyDrive/Cerebrasensecloud/OASIS-1 \
  --run-name oasis_colab_full_v3_auroc_monitor \
  --epochs 28 \
  --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --image-size 64 64 64 \
  --seed 42 \
  --split-seed 42 \
  --early-stopping-monitor val_auroc \
  --early-stopping-mode max \
  --early-stopping-patience 6 \
  --stage-oasis-to-local \
  --resume-if-available \
  --calibrate-threshold \
  --promote
```

## Recommended Kaggle Flow

Use [kaggle_train.ipynb](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/kaggle_train.ipynb).

That notebook already handles:

- Drive mount
- fresh clone
- dependency install
- persistent runtime roots under `backend_runtime/`
- local staging for the Kaggle upload bundle
- manifest and split rebuilds
- safe resume or reuse of the same `RUN_NAME`
- summary inspection from `colab_run_summary.json`
- optional single-image inference from the saved checkpoint

If you prefer the script directly, the main entrypoint is:

```bash
cd /content/Cerebrasense-/alz_backend
python scripts/train_kaggle_colab.py \
  --project-root /content/Cerebrasense-/alz_backend \
  --runtime-root /content/drive/MyDrive/Cerebrasensecloud/backend_runtime \
  --kaggle-source-dir /content/drive/MyDrive/Cerebrasensecloud/kaggle_alz_upload_bundle \
  --run-name kaggle_colab_baseline_v1 \
  --epochs 12 \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --image-size-2d 224 224 \
  --image-size-3d 128 128 128 \
  --seed 42 \
  --split-random-state 42 \
  --train-fraction 0.7 \
  --val-fraction 0.15 \
  --test-fraction 0.15 \
  --early-stopping-monitor val_macro_f1 \
  --early-stopping-mode max \
  --early-stopping-patience 4 \
  --stage-kaggle-to-local \
  --resume-if-available
```

## Kaggle Upload Reminder

Before running the Kaggle notebook, upload the full bundle folder described in [kaggle_drive_upload_checklist.md](/C:/Users/Nguyen%20Quang%20Minh/OneDrive/Desktop/Cerebrasense/archive%20(1)/alz_backend/docs/kaggle_drive_upload_checklist.md).

The safest source is the generated local bundle:

```text
alz_backend/outputs/exports/kaggle_alz_upload_bundle/
```

Upload the whole `kaggle_alz_upload_bundle/` folder to Drive, not loose files from inside it.

## Important Runtime Rules

- Keep the same `RUN_NAME` only when you want to resume or reuse that exact experiment.
- Change `RUN_NAME` when you want a genuinely new experiment.
- After a Colab restart, `/content` is gone, but `backend_runtime/` on Drive still keeps your persisted manifests, metrics, and checkpoints.
- `backend_runtime/` is the canonical cloud source of truth for promoted OASIS runs.
- OASIS and Kaggle remain separate evidence branches.
- OASIS-2 remains a preparation and readiness branch, not a supervised training branch.

## Scope Reminder

- OASIS is the primary supervised 3D structural MRI branch.
- Kaggle is the secondary 2D comparison branch.
- OASIS-2 is for structural and longitudinal preparation until labels and metadata are ready.
- Do not merge Kaggle and OASIS evidence implicitly.
