# Colab Quickstart for `Cerebrasensecloud`

Use this when your datasets live in Google Drive under:

```text
/content/drive/MyDrive/Cerebrasensecloud/
```

Current detected dataset roots:

- OASIS-1: `/content/drive/MyDrive/Cerebrasensecloud/OASIS-1`
- OASIS-2: `/content/drive/MyDrive/Cerebrasensecloud/OASIS-2`
- Kaggle: `/content/drive/MyDrive/Cerebrasensecloud/Kaggle`

The repo should be cloned separately inside Colab, then outputs will be saved under the repo's `outputs/` folder.

## Cell 1: Mount Drive and Define Paths

```python
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

DATA_ROOT = Path("/content/drive/MyDrive/Cerebrasensecloud")
OASIS1 = DATA_ROOT / "OASIS-1"
OASIS2 = DATA_ROOT / "OASIS-2"
KAGGLE = DATA_ROOT / "Kaggle"

for name, path in {
    "DATA_ROOT": DATA_ROOT,
    "OASIS1": OASIS1,
    "OASIS2": OASIS2,
    "KAGGLE": KAGGLE,
}.items():
    print(name, path.exists(), path)
```

## Cell 2: Clone the Repo and Install Dependencies

Replace `YOUR_GITHUB_USERNAME` only if needed.

```bash
cd /content
rm -rf Cerebrasense-
git clone https://github.com/Billrichard209/Cerebrasense-.git
cd Cerebrasense-/alz_backend
python -m pip install --upgrade pip
pip install -r requirements-colab.txt
```

## Cell 3: Export Runtime Paths

```python
import os

os.environ["ALZ_OASIS_SOURCE_DIR"] = str(OASIS1)
os.environ["ALZ_OASIS2_SOURCE_DIR"] = str(OASIS2)
os.environ["ALZ_KAGGLE_SOURCE_DIR"] = str(KAGGLE)

print(os.environ["ALZ_OASIS_SOURCE_DIR"])
print(os.environ["ALZ_OASIS2_SOURCE_DIR"])
print(os.environ["ALZ_KAGGLE_SOURCE_DIR"])
```

## Cell 4: Sanity-Check the Datasets

```bash
cd /content/Cerebrasense-/alz_backend
python scripts/build_oasis_manifest.py --output-format csv
python scripts/build_kaggle_manifest.py --output-format csv
python scripts/check_oasis2_readiness.py --source-root "/content/drive/MyDrive/Cerebrasensecloud/OASIS-2"
python scripts/build_oasis2_inventory.py --source-root "/content/drive/MyDrive/Cerebrasensecloud/OASIS-2"
python scripts/build_oasis2_session_manifest.py --source-root "/content/drive/MyDrive/Cerebrasensecloud/OASIS-2"
```

## Cell 5: Start OASIS-1 Training

```bash
cd /content/Cerebrasense-/alz_backend
python scripts/train_oasis_colab.py \
  --oasis-source-dir "/content/drive/MyDrive/Cerebrasensecloud/OASIS-1" \
  --run-name oasis_colab_scope_v1 \
  --epochs 20 \
  --evaluate-splits val test
```

## Optional: Resume or Run a Smaller Trial

For a smaller first run:

```bash
cd /content/Cerebrasense-/alz_backend
python scripts/train_oasis_colab.py \
  --oasis-source-dir "/content/drive/MyDrive/Cerebrasensecloud/OASIS-1" \
  --run-name oasis_colab_smoke \
  --epochs 2 \
  --batch-size 2 \
  --image-size 64 64 64 \
  --evaluate-splits val
```

## Notes

- Keep OASIS-1 as the primary supervised 3D branch.
- Keep Kaggle fully separate as the 2D comparison branch.
- Treat OASIS-2 as structural/longitudinal preparation only until labels/metadata are available.
- If Colab memory is tight, reduce `--batch-size` from `4` to `2`.
