# FastAPI Backend Usage

This API is for OASIS-first structural brain MRI decision-support research
workflows. It does not diagnose disease and keeps OASIS-1 and Kaggle data
separate.

The current backend scope is best understood as dementia-related structural MRI
analysis and longitudinal tracking research, not broad brain-disease detection.

## Run Locally

From `alz_backend`:

```powershell
cmd /c run_api.cmd
```

Or directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

For a one-command local demo artifact bundle that exercises the main API surface without manually calling each route, use:

```powershell
cmd /c build_oasis_demo_bundle.cmd --scan-path "C:\path\to\scan.hdr" --device cpu
```

## Optional API Key

For local development, routes are open unless an API key is configured. To
enable starter auth:

```powershell
$env:ALZ_API_KEY="secret-dev-key"
```

Then send:

```text
X-API-Key: secret-dev-key
```

## Example Requests

Health:

```powershell
curl http://127.0.0.1:8000/health
```

Model metadata:

```powershell
curl http://127.0.0.1:8000/models/oasis/metadata
```

Predict an existing scan path:

```powershell
curl -X POST http://127.0.0.1:8000/predict/scan `
  -H "Content-Type: application/json" `
  -d "{\"scan_path\":\"C:\\path\\to\\scan.hdr\",\"checkpoint_path\":\"outputs\\runs\\oasis\\oasis_experiment_tiny\\checkpoints\\best_model.pt\",\"config_path\":\"configs\\oasis_transforms.yaml\",\"threshold\":0.5,\"device\":\"cpu\"}"
```

Predict a raw NIfTI upload without multipart dependencies:

```powershell
curl -X POST "http://127.0.0.1:8000/predict/scan/upload?file_name=scan.nii.gz&checkpoint_path=outputs\runs\oasis\oasis_experiment_tiny\checkpoints\best_model.pt" `
  -H "Content-Type: application/octet-stream" `
  --data-binary "@C:\path\to\scan.nii.gz"
```

Explain a scan:

```powershell
curl -X POST http://127.0.0.1:8000/explain/scan `
  -H "Content-Type: application/json" `
  -d "{\"scan_path\":\"C:\\path\\to\\scan.hdr\",\"checkpoint_path\":\"outputs\\runs\\oasis\\oasis_experiment_tiny\\checkpoints\\best_model.pt\",\"preprocessing_config_path\":\"configs\\oasis_transforms.yaml\",\"target_layer\":\"auto\",\"slice_axis\":\"axial\",\"slice_indices\":[16,32,48],\"device\":\"cpu\"}"
```

Generate a longitudinal structural proxy report:

```powershell
curl -X POST http://127.0.0.1:8000/longitudinal/report `
  -H "Content-Type: application/json" `
  -d "{\"subject_id\":\"OAS1_0001\",\"output_name\":\"OAS1_0001_api_longitudinal\"}"
```

Generate a timeline-ready longitudinal trend report from explicit scan history:

```powershell
curl -X POST http://127.0.0.1:8000/longitudinal/report `
  -H "Content-Type: application/json" `
  -d "{\"subject_id\":\"OAS1_0002\",\"output_name\":\"OAS1_0002_timeline\",\"records\":[{\"subject_id\":\"OAS1_0002\",\"session_id\":\"OAS1_0002_MR1\",\"visit_order\":1,\"scan_timestamp\":\"2001-01-01\",\"source_path\":\"scan1.hdr\",\"dataset\":\"oasis1\",\"volumetric_features\":{\"left_hippocampus_volume_mm3\":3200.0},\"model_probabilities\":{\"ad_like_probability\":0.20}},{\"subject_id\":\"OAS1_0002\",\"session_id\":\"OAS1_0002_MR2\",\"visit_order\":2,\"scan_timestamp\":\"2002-01-01\",\"source_path\":\"scan2.hdr\",\"dataset\":\"oasis1\",\"volumetric_features\":{\"left_hippocampus_volume_mm3\":3008.0},\"model_probabilities\":{\"ad_like_probability\":0.32}}]}"
```

Legacy-compatible structural endpoint:

```powershell
curl "http://127.0.0.1:8000/longitudinal/oasis/OAS1_0001/volumetrics"
```

## Notes

- The upload endpoint supports `.nii` and `.nii.gz` only. Use the path-based
  endpoint for Analyze `.hdr` / `.img` pairs so both files remain available.
- Prediction and explanation outputs are saved under `outputs/predictions/`
  and `outputs/visualizations/explanations/`.
- Medical wording must remain decision-support only.
- OASIS is the primary 3D reference branch. Kaggle remains a separate 2D
  comparison branch.
