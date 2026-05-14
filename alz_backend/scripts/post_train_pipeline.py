"""
Post-Training Pipeline: Evaluate -> Audit -> Export -> Deploy
Run this after a Colab training run completes.

Usage:
  python scripts/post_train_pipeline.py --run-name oasis2_monotonic_v2
"""
import argparse
import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_step(name, cmd):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Post-training automation pipeline.")
    parser.add_argument("--run-name", required=True, help="Training run name (e.g. oasis2_monotonic_v2)")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    args = parser.parse_args()

    run_dir = ROOT / "outputs" / "runs" / "oasis2" / args.run_name
    checkpoint = run_dir / "checkpoints" / "best_model.pt"
    eval_dir = run_dir / "evaluation" / "post_train_test_best_model"
    predictions_csv = eval_dir / "predictions.csv"
    audit_json = ROOT / "outputs" / "reports" / "longitudinal" / f"audit_{args.run_name}.json"
    onnx_output = ROOT / f"best_model_{args.run_name}.onnx"

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        print("Copy it from Google Drive first.")
        sys.exit(1)

    # Step 1: Evaluate
    if not args.skip_eval:
        ok = run_step("Evaluate Model", [
            sys.executable, "scripts/evaluate_oasis2_candidate.py",
            "--run-name", args.run_name,
            "--checkpoint-path", str(checkpoint),
            "--device", args.device,
        ])
        if not ok:
            print("Evaluation failed. Stopping.")
            sys.exit(1)

    # Step 2: Audit Temporal Paradoxes
    if predictions_csv.exists():
        ok = run_step("Audit Temporal Paradoxes", [
            sys.executable, "scripts/audit_temporal_paradoxes.py",
            "--predictions-csv", str(predictions_csv),
            "--output-json", str(audit_json),
        ])
        if ok and audit_json.exists():
            with open(audit_json) as f:
                audit = json.load(f)
            total = audit.get("total_transitions", 0)
            paradoxes = audit.get("paradox_count", 0)
            rate = (paradoxes / total * 100) if total > 0 else 0
            print(f"\n  RESULT: {paradoxes}/{total} paradoxes ({rate:.1f}%)")
            if paradoxes == 0:
                print("  PERFECT SCORE - No paradoxes detected!")
    else:
        print(f"  Predictions CSV not found at {predictions_csv}")
        print("  Skipping audit.")

    # Step 3: Export to ONNX
    ok = run_step("Export to ONNX", [
        sys.executable, "scripts/export_onnx.py",
        "--checkpoint", str(checkpoint),
        "--output", str(onnx_output),
    ])

    # Step 4: Summary
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Checkpoint:  {checkpoint}")
    print(f"  Predictions: {predictions_csv}")
    print(f"  Audit:       {audit_json}")
    print(f"  ONNX Model:  {onnx_output}")
    print(f"\n  Next: Restart the dashboard server to use the new model.")
    print(f"    python dashboard/serve.py")

if __name__ == "__main__":
    main()
