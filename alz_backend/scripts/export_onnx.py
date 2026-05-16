"""ONNX Export Utility for MRI Models.

Converts a trained PyTorch .pt checkpoint into a .onnx file for 
high-performance deployment and reduced VRAM usage.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.onnx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import load_oasis_model_config, build_model
from src.evaluation.oasis_run import load_oasis_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Export MRI model to ONNX.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to .pt checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="Output .onnx path.")
    parser.add_argument("--image-size", type=int, nargs=3, default=[96, 96, 96], help="Input volume dimensions.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    output_path = args.output or args.checkpoint.with_suffix(".onnx")
    
    print(f"Loading PyTorch model from {args.checkpoint}...")
    checkpoint = load_oasis_checkpoint(args.checkpoint, device="cpu")
    
    # We try to find the model config relative to the checkpoint
    config_path = args.checkpoint.parents[1] / "model_config.yaml"
    model_cfg = load_oasis_model_config(config_path if config_path.exists() else None)
    
    model = build_model(model_cfg)
    model.load_state_dict(checkpoint.model_state_dict)
    model.eval()

    # Create dummy input for tracing (Batch, Channel, D, H, W)
    dummy_mri = torch.randn(1, 1, *args.image_size)
    
    # Multimodal handling
    if hasattr(model, "tabular_mlp"):
        dummy_clinical = torch.randn(1, 3)  # [Age, Sex, MMSE]
        dummy_input = (dummy_mri, dummy_clinical)
        input_names = ["input_mri", "input_clinical"]
        dynamic_axes = {
            "input_mri": {0: "batch_size"},
            "input_clinical": {0: "batch_size"},
            "risk_logits": {0: "batch_size"}
        }
    else:
        dummy_input = dummy_mri
        input_names = ["input_mri"]
        dynamic_axes = {
            "input_mri": {0: "batch_size"},
            "risk_logits": {0: "batch_size"}
        }

    print(f"Exporting to ONNX: {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["risk_logits"],
        dynamic_axes=dynamic_axes
    )

    print("✅ Export successful!")
    print(f"ONNX Model saved to: {output_path}")

if __name__ == "__main__":
    main()
