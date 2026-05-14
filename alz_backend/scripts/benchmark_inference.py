"""
Benchmarking script to compare PyTorch vs ONNX inference speed.
"""
import time
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Paths
ROOT = Path(r"c:\Users\Nguyen Quang Minh\OneDrive\Desktop\Cerebrasense\archive (1)\alz_backend")
PT_PATH = ROOT / "outputs/runs/oasis2/oasis2_colab_improved_v1/checkpoints/best_model.pt"
ONNX_PATH = ROOT / "best_model.onnx"

def benchmark():
    if not PT_PATH.exists() or not ONNX_PATH.exists():
        print("Model files not found. Ensure export_onnx.py has been run.")
        return

    # Create dummy MRI input (1, 1, 96, 96, 96)
    dummy_input = torch.randn(1, 1, 96, 96, 96)
    input_np = dummy_input.numpy()

    print("\n--- STARTING BENCHMARK (RTX 2050) ---")

    # 1. PyTorch Benchmark
    print("Loading PyTorch Model...")
    from monai.networks.nets import Classifier
    model_pt = Classifier(
        in_shape=(1, 96, 96, 96),
        classes=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2, 2),
    ).cuda()
    
    # Warmup
    for _ in range(5): _ = model_pt(dummy_input.cuda())
    
    start = time.time()
    for _ in range(20):
        with torch.no_grad():
            _ = model_pt(dummy_input.cuda())
    pt_time = (time.time() - start) / 20
    print(f"PyTorch Average Latency: {pt_time*1000:.2f}ms")

    # 2. ONNX Benchmark
    print("Loading ONNX Session...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(ONNX_PATH), providers=providers)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(5): _ = session.run(None, {input_name: input_np})

    start = time.time()
    for _ in range(20):
        _ = session.run(None, {input_name: input_np})
    onnx_time = (time.time() - start) / 20
    print(f"ONNX Average Latency:    {onnx_time*1000:.2f}ms")

    speedup = pt_time / onnx_time
    print(f"\nTotal Speedup: {speedup:.2f}x faster")
    print("---------------------------------------\n")

if __name__ == "__main__":
    benchmark()
