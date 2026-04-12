# Kaggle Drive Upload Checklist

Use this file when preparing the Kaggle Alzheimer dataset for Google Drive.

This dataset is part of the project's **secondary 2D comparison branch**.
It should stay fully separate from OASIS.

## Upload These Folders

Keep the original folder names if possible:

```text
OriginalDataset/
AugmentedAlzheimerDataset/
```

## Recommended Approach

Do not manually assemble the Kaggle package if you do not need to.

Build the portable bundle instead:

```powershell
.\.venv\Scripts\python.exe alz_backend\scripts\build_kaggle_upload_bundle.py --materialize-mode hardlink
```

That produces a ready-to-upload folder at:

```text
alz_backend/outputs/exports/kaggle_alz_upload_bundle/
```

The bundle keeps the two Kaggle dataset roots plus the manifest and split references together.

## Recommended Google Drive Layout

```text
MyDrive/
  cerebrasense/
    data/
      kaggle_alz/
        raw/
          OriginalDataset/
          AugmentedAlzheimerDataset/
```

## What To Upload

Upload:

- the whole generated `kaggle_alz_upload_bundle/` folder

If you skip the bundle approach, then upload:

- the full `OriginalDataset/` folder
- the full `AugmentedAlzheimerDataset/` folder

Keep class subfolders and image files exactly as they are.

## What Not To Upload

Do **not** upload these as part of the Kaggle dataset package:

- `.venv/`
- `outputs/`
- `__pycache__/`
- backend source code
- reports
- checkpoints
- unrelated CSV files
- zipped duplicates after extraction

## If You Want To Zip First

These zip names are fine:

```text
kaggle_alz_original.zip
kaggle_alz_augmented.zip
```

After uploading to Drive, extract them so Colab can read the image folders directly.

## Expected Colab Paths

If you follow the recommended layout, the Kaggle paths will be:

```text
/content/drive/MyDrive/cerebrasense/data/kaggle_alz/upload_bundle/OriginalDataset
/content/drive/MyDrive/cerebrasense/data/kaggle_alz/upload_bundle/AugmentedAlzheimerDataset
```

If you use the bundle, point `ALZ_KAGGLE_SOURCE_DIR` at the bundle root itself:

```text
/content/drive/MyDrive/cerebrasense/data/kaggle_alz/upload_bundle
```

## Scope Reminder

- Kaggle is `2d_slices`
- Kaggle is useful for comparison experiments
- Kaggle is **not** a replacement for the OASIS 3D evidence path
- Do not merge Kaggle and OASIS implicitly

## Quick Check Before Upload

Make sure:

- `OriginalDataset/` exists
- `AugmentedAlzheimerDataset/` exists
- class folders are still inside each dataset folder
- image files are still inside the class folders
- no backend outputs or virtualenv files are mixed in
