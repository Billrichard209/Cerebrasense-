# Google Colab Training Setup

This backend can be trained in Google Colab so the local laptop does not carry
the full 3D MONAI training load.

## Recommended Colab Flow

1. Open a GPU runtime in Colab.
2. Put the repository folder and the OASIS data in Google Drive.
3. Mount Drive inside Colab.
4. Install project dependencies from `requirements-colab.txt`.
5. Run the Colab training wrapper with the Drive path to OASIS.

## Minimal Colab Commands

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content/drive/MyDrive/YOUR_PATH/alz_backend
pip install -r requirements-colab.txt
python scripts/train_oasis_colab.py \
  --mount-drive \
  --oasis-source-dir "/content/drive/MyDrive/YOUR_PATH/OASIS" \
  --run-name oasis_colab_v1
```

## Colab Notes

- `requirements-colab.txt` intentionally does not pin `torch` or `torchvision`.
  Colab usually provides a CUDA-enabled PyTorch already.
- The default Colab config uses:
  - `device=cuda`
  - `mixed_precision=true`
  - `batch_size=4`
  - `image_size=[64, 64, 64]`
  - `num_workers=2`
- If Colab memory is tight, lower `batch_size` to `2`.
- Keep OASIS and Kaggle separate in Colab exactly as in local training.
- Outputs are still written under `outputs/runs/oasis/...` inside the repo.

## Suggested Drive Layout

```text
MyDrive/
  alz_backend/
  OASIS/
```

## Decision-Support Reminder

This training workflow is for research and clinical decision support only. It is
not diagnosis software.
