"""Build a compact hard-case benchmark summary from the escalated OASIS review set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import AppSettings, get_app_settings  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402


def _safe_name(value: str) -> str:
    """Return a path-safe name."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def build_oasis_hard_case_benchmark(
    *,
    settings: AppSettings | None = None,
    handoff_name: str = "oasis_specialist_handoff_pack",
    output_name: str = "oasis_hard_case_benchmark",
) -> dict[str, Any]:
    """Build a compact benchmark summary from the escalated OASIS hard cases."""

    resolved_settings = settings or get_app_settings()
    safe_handoff_name = _safe_name(handoff_name)
    safe_output_name = _safe_name(output_name)

    handoff_root = resolved_settings.outputs_root / "reports" / "review" / safe_handoff_name
    escalated_cases_csv = handoff_root / "escalated_cases.csv"
    if not escalated_cases_csv.exists():
        raise FileNotFoundError(f"Escalated cases CSV not found: {escalated_cases_csv}")

    frame = pd.read_csv(escalated_cases_csv, dtype=str).fillna("")
    if "probability_score" in frame.columns:
        frame["probability_score"] = frame["probability_score"].astype(float)
        frame["distance_from_boundary"] = (frame["probability_score"] - 0.5).abs()
    else:
        frame["distance_from_boundary"] = float("nan")

    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "benchmark" / safe_output_name)
    benchmark_cases_csv = output_root / "hard_case_benchmark_cases.csv"
    frame.to_csv(benchmark_cases_csv, index=False)

    priority_counts = frame.get("reviewer_priority", pd.Series(dtype="string")).astype(str).value_counts(dropna=False).to_dict()
    label_counts = frame.get("label_name", pd.Series(dtype="string")).astype(str).value_counts(dropna=False).to_dict()
    confidence_counts = frame.get("confidence_level", pd.Series(dtype="string")).astype(str).value_counts(dropna=False).to_dict()

    summary = {
        "handoff_name": safe_handoff_name,
        "output_name": safe_output_name,
        "handoff_root": str(handoff_root),
        "output_root": str(output_root),
        "hard_case_count": int(len(frame)),
        "high_priority_count": int((frame.get("reviewer_priority", pd.Series(dtype="string")).astype(str) == "high").sum()),
        "label_counts": label_counts,
        "confidence_counts": confidence_counts,
        "priority_counts": priority_counts,
        "mean_probability": float(frame["probability_score"].mean()) if len(frame) else None,
        "mean_distance_from_boundary": float(frame["distance_from_boundary"].mean()) if len(frame) else None,
        "closest_cases": frame.sort_values(by=["distance_from_boundary", "probability_score"], ascending=[True, False], kind="stable")
        [
            [column for column in ["rank", "subject_id", "session_id", "label_name", "probability_score", "reviewer_priority"] if column in frame.columns]
        ]
        .head(5)
        .to_dict(orient="records"),
        "notes": [
            "This benchmark tracks the model's most uncertain OASIS cases without pretending they are adjudicated labels.",
            "Use this set to compare future OASIS runs on uncertainty concentration and review burden, not on supervised correctness.",
            "These cases remain an uncertainty benchmark until qualified review becomes available.",
        ],
        "benchmark_cases_csv": str(benchmark_cases_csv),
    }
    summary_json_path = output_root / "hard_case_benchmark_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# OASIS Hard-Case Benchmark",
        "",
        f"- hard_case_count: {summary['hard_case_count']}",
        f"- high_priority_count: {summary['high_priority_count']}",
        f"- mean_probability: {summary['mean_probability']:.3f}" if summary["mean_probability"] is not None else "- mean_probability: n/a",
        f"- mean_distance_from_boundary: {summary['mean_distance_from_boundary']:.3f}" if summary["mean_distance_from_boundary"] is not None else "- mean_distance_from_boundary: n/a",
        f"- benchmark_cases_csv: {benchmark_cases_csv}",
        "",
        "## Closest Cases",
        "",
    ]
    if summary["closest_cases"]:
        for case in summary["closest_cases"]:
            md_lines.append(
                f"- {case.get('rank')}. {case.get('subject_id')} / {case.get('session_id')} "
                f"({case.get('label_name')}, p={float(case.get('probability_score')):.3f}, priority={case.get('reviewer_priority')})"
            )
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Notes", ""])
    md_lines.extend(f"- {note}" for note in summary["notes"])
    summary_md_path = output_root / "hard_case_benchmark_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary["summary_json_path"] = str(summary_json_path)
    summary["summary_md_path"] = str(summary_md_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build an OASIS hard-case benchmark summary from the escalated review set.")
    parser.add_argument("--handoff-name", type=str, default="oasis_specialist_handoff_pack")
    parser.add_argument("--output-name", type=str, default="oasis_hard_case_benchmark")
    return parser


def main() -> None:
    """Build the benchmark summary and print a compact summary."""

    args = build_parser().parse_args()
    summary = build_oasis_hard_case_benchmark(
        handoff_name=args.handoff_name,
        output_name=args.output_name,
    )
    print(f"output_root={summary['output_root']}")
    print(f"hard_case_count={summary['hard_case_count']}")
    print("summary=" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
