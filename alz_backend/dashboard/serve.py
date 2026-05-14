"""
CerebraSense Clinical Dashboard Server
Serves the dashboard HTML and exposes a JSON API from the predictions CSV.
Usage: python dashboard/serve.py
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Add project root to sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import cgi
from src.inference.pipeline import predict_scan, PredictScanOptions
from src.configs.runtime import get_app_settings

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DASHBOARD_DIR = Path(__file__).parent

# Try to find the best available predictions CSV (OASIS-2 preferred)
CANDIDATE_CSVS = [
    ROOT / "outputs/runs/oasis2/oasis2_colab_improved_v1/evaluation/post_train_test_best_model/predictions.csv",
    ROOT / "outputs/runs/oasis2/oasis2_bias_stability_v1/evaluation/post_train_val_best_model/predictions.csv",
    ROOT / "outputs/runs/oasis2/oasis2_bias_stability_v1/evaluation/post_train_test_best_model/predictions.csv",
]

AUDIT_JSON = ROOT / "outputs/reports/longitudinal/audit_consistent_final.json"


def find_predictions_csv():
    for p in CANDIDATE_CSVS:
        if p.exists():
            return p
    return None


def ema_smooth(scores: list, alpha: float = 0.4) -> list:
    smoothed = []
    current = scores[0]
    for s in scores:
        current = alpha * s + (1 - alpha) * current
        smoothed.append(round(current, 4))
    return smoothed


def detect_paradoxes(scores: list, epsilon: float = 0.05) -> list:
    """Return indices where a significant drop occurred."""
    paradox_indices = []
    for i in range(1, len(scores)):
        if scores[i] < scores[i - 1] - epsilon:
            paradox_indices.append(i)
    return paradox_indices


def load_data():
    # Define primary and secondary comparison runs
    RUNS = {
        "Consistent (V2)": ROOT / "outputs/runs/oasis2/oasis2_colab_improved_v1/evaluation/post_train_test_best_model/predictions.csv",
        "Baseline (V1)": ROOT / "outputs/runs/oasis2/oasis2_bias_stability_v1/evaluation/post_train_test_best_model/predictions.csv",
    }

    run_data = {}
    for name, path in RUNS.items():
        if path.exists():
            run_data[name] = pd.read_csv(path)

    if not run_data:
        return None, "No prediction CSVs found in outputs/"

    # Use the primary run for the subject list
    primary_name = "Consistent (V2)" if "Consistent (V2)" in run_data else list(run_data.keys())[0]
    df_primary = run_data[primary_name]
    
    longitudinal = df_primary[df_primary["meta_subject_id"].str.startswith("OAS2_", na=False)].copy()
    if longitudinal.empty:
        longitudinal = df_primary.copy()

    subjects = []
    for subj_id, group in longitudinal.groupby("meta_subject_id"):
        group = group.sort_values("meta_session_id").reset_index(drop=True)
        raw_scores = group["probability_class_1"].tolist()
        smoothed = ema_smooth(raw_scores)
        paradox_idx = detect_paradoxes(smoothed)
        
        # Try to get baseline scores for comparison
        comparison_scores = []
        if "Baseline (V1)" in run_data:
            b_df = run_data["Baseline (V1)"]
            b_subj = b_df[b_df["meta_subject_id"] == subj_id].sort_values("meta_session_id")
            if not b_subj.empty:
                comparison_scores = [round(s, 4) for s in b_subj["probability_class_1"].tolist()]

        final_risk = smoothed[-1]
        subjects.append({
            "subject_id": subj_id,
            "visits": group["meta_session_id"].tolist(),
            "raw_scores": [round(s, 4) for s in raw_scores],
            "smoothed_scores": smoothed,
            "comparison_scores": comparison_scores,
            "paradox_indices": paradox_idx,
            "true_label": group["true_label_name"].iloc[-1],
            "final_risk": round(final_risk, 4),
            "status": "High Risk" if final_risk >= 0.65 else "Low Risk",
            "clinical_flag": "Paradox Detected" if paradox_idx else "Stable Trend",
            "num_visits": len(raw_scores),
        })

    subjects.sort(key=lambda x: x["final_risk"], reverse=True)

    return {
        "subjects": subjects,
        "summary": {
            "total_subjects": len(subjects),
            "runs_loaded": list(run_data.keys()),
        }
    }, None


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"  [{self.command}] {self.path}")

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str):
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/" or path == "/index.html":
            html_file = DASHBOARD_DIR / "index.html"
            if html_file.exists():
                self.send_file(html_file, "text/html; charset=utf-8")
            else:
                self.send_json({"error": "index.html not found"}, 404)

        elif path == "/api/data":
            data, err = load_data()
            if err:
                self.send_json({"error": err}, 500)
            else:
                self.send_json(data)

        elif path == "/api/health":
            self.send_json({"status": "ok", "csv_found": find_predictions_csv() is not None})

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/api/predict":
            try:
                # Handle multipart upload
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type']}
                )

                if 'file' not in form:
                    self.send_json({"error": "No file uploaded"}, 400)
                    return

                file_item = form['file']
                if not file_item.filename:
                    self.send_json({"error": "No filename"}, 400)
                    return

                # Save temp file
                temp_dir = ROOT / "outputs" / "tmp" / "uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / file_item.filename
                temp_path.write_bytes(file_item.file.read())

                # Run Inference using ONNX (preferred)
                onnx_path = ROOT / "best_model.onnx"
                pt_path = ROOT / "outputs/runs/oasis2/oasis2_colab_improved_v1/checkpoints/best_model.pt"
                checkpoint = onnx_path if onnx_path.exists() else pt_path

                if not checkpoint.exists():
                    self.send_json({"error": "No model found. Export ONNX first."}, 500)
                    return

                print(f"  [Predict] Running inference for {file_item.filename} using {checkpoint.name}...")
                
                result = predict_scan(
                    scan_path=str(temp_path),
                    checkpoint_path=str(checkpoint),
                    options=PredictScanOptions(
                        output_name=f"upload_{file_item.filename}",
                        device="cpu" # Dashboard usually on CPU for safety
                    ),
                    settings=get_app_settings()
                )

                self.send_json(result)

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                self.send_json({"error": str(e)}, 500)
        else:
            self.send_json({"error": "Not found"}, 404)


def main():
    port = int(os.environ.get("PORT", 8765))
    server = HTTPServer(("localhost", port), DashboardHandler)
    print(f"\nCerebraSense Dashboard running at: http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
