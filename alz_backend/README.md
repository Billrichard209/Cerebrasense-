# CerebraSense | Longitudinal Alzheimer's Intelligence 🧠🛡️

CerebraSense is a production-grade clinical intelligence system designed for longitudinal MRI analysis. It leverages deep learning and biological constraints to provide stable, paradox-free risk assessments for Alzheimer's progression.

## 🏗️ The 4 Pillars of Clinical Intelligence

1.  **Biological Constraint (Training)**: Integrated `MonotonicityLoss` to ensure that risk scores only increase or stabilize over time, mirroring the biological reality of neurodegeneration.
2.  **Temporal Paradox Audit (Validation)**: A rigorous validation framework that detects impossible "risk drops" across patient visits, ensuring clinical reliability.
3.  **Longitudinal API (Smoothing)**: A FastAPI-based service that applies Exponential Moving Average (EMA) smoothing to raw model outputs, providing de-noised clinical trends.
4.  **ONNX Optimization (Performance)**: High-performance inference engine optimized for local deployment (RTX 2050), reducing model size to 45MB with ~250ms latency.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- CUDA-enabled GPU (RTX 2050 or better recommended)
- `onnxruntime-gpu`, `monai`, `torch`

### 2. Launch the Clinical Dashboard
```powershell
cd alz_backend
python dashboard/serve.py
```
Open **http://localhost:8765** to view the patient risk timelines and upload new scans.

### 3. Run the Post-Training Pipeline
After training a new model on Google Colab:
```powershell
python scripts/post_train_pipeline.py --run-name oasis2_monotonic_v2
```
This automatically evaluates, audits, and exports the model for production use.

---

## 📊 Performance & Accuracy
- **Model**: Optimized 3D ResNet/DenseNet via MONAI.
- **Size**: 135MB (PyTorch) → 45MB (ONNX).
- **Latency**: ~250ms per volume on CPU / <50ms on GPU.
- **Dataset**: OASIS-2 Longitudinal (Long-term progression tracking).

## 🛡️ Clinical Disclaimer
This system is a decision-support tool and does not provide a primary diagnosis. All assessments should be reviewed by a qualified clinical professional.

---
*Built with ❤️ for the future of neuro-imaging intelligence.*
