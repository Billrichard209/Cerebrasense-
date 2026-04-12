"""Scope-aware code and dataset planning for GitHub, Drive, and Colab workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    """Return paths in stable order without duplicates."""

    seen: set[str] = set()
    unique_paths: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def _relative_label(path: Path, workspace_root: Path) -> str:
    """Render a path relative to the workspace root when possible."""

    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _dataset_section(
    *,
    workspace_root: Path,
    name: str,
    role: str,
    local_candidates: list[Path],
    recommended_drive_root: str,
    notes: list[str],
) -> dict[str, Any]:
    """Build one dataset planning section."""

    deduped_candidates = _dedupe_paths(local_candidates)
    existing_paths = [path for path in deduped_candidates if path.exists()]
    return {
        "dataset": name,
        "role": role,
        "present": bool(existing_paths),
        "local_candidates": [_relative_label(path, workspace_root) for path in deduped_candidates],
        "existing_local_paths": [_relative_label(path, workspace_root) for path in existing_paths],
        "recommended_drive_root": recommended_drive_root,
        "notes": notes,
    }


def build_scope_data_plan(settings: AppSettings | None = None) -> dict[str, Any]:
    """Build a practical code/data plan for the narrowed OASIS-first project scope."""

    resolved_settings = settings or get_app_settings()
    workspace_root = resolved_settings.workspace_root
    project_root = resolved_settings.project_root
    gitignore_path = workspace_root / ".gitignore"

    oasis_candidates = [
        resolved_settings.oasis_source_root,
        workspace_root / "OASIS",
        workspace_root / "datasets" / "oasis1" / "raw",
    ]
    kaggle_candidates = [
        workspace_root / "OriginalDataset",
        workspace_root / "AugmentedAlzheimerDataset",
        workspace_root / "datasets" / "kaggle_alz" / "raw" / "OriginalDataset",
        workspace_root / "datasets" / "kaggle_alz" / "raw" / "AugmentedAlzheimerDataset",
    ]
    oasis2_candidates = [
        resolved_settings.collection_root / "OASIS2",
        resolved_settings.collection_root / "OAS2_RAW_PART1",
        resolved_settings.collection_root / "OAS2_RAW_PART2",
        workspace_root / "OASIS2",
        workspace_root / "datasets" / "oasis2" / "raw",
    ]

    datasets = {
        "oasis1": _dataset_section(
            workspace_root=workspace_root,
            name="oasis1",
            role="primary_3d_evidence_track",
            local_candidates=oasis_candidates,
            recommended_drive_root="cerebrasense/data/oasis1/raw",
            notes=[
                "Keep raw OASIS-1 scans and metadata together.",
                "Keep generated manifests and split CSVs in the backend repo for reproducibility.",
            ],
        ),
        "kaggle_alz": _dataset_section(
            workspace_root=workspace_root,
            name="kaggle_alz",
            role="secondary_2d_comparison_branch",
            local_candidates=kaggle_candidates,
            recommended_drive_root="cerebrasense/data/kaggle_alz/raw",
            notes=[
                "Keep Kaggle fully separate from OASIS.",
                "Store OriginalDataset and AugmentedAlzheimerDataset under the same raw Kaggle branch.",
            ],
        ),
        "oasis2": _dataset_section(
            workspace_root=workspace_root,
            name="oasis2",
            role="future_longitudinal_extension",
            local_candidates=oasis2_candidates,
            recommended_drive_root="cerebrasense/data/oasis2/raw",
            notes=[
                "Reserve this slot for future longitudinal onboarding.",
                "Do not mix OASIS-2 into OASIS-1 until a dedicated adapter exists.",
            ],
        ),
    }

    include_paths = [
        "alz_backend/src/",
        "alz_backend/scripts/",
        "alz_backend/configs/",
        "alz_backend/docs/",
        "alz_backend/notebooks/",
        "alz_backend/requirements.txt",
        "alz_backend/requirements-colab.txt",
        ".gitignore",
    ]
    exclude_paths = [
        ".venv/",
        "alz_backend/outputs/",
        "alz_backend/storage/*.sqlite3",
        "OriginalDataset/",
        "AugmentedAlzheimerDataset/",
        "OASIS/",
        "OASIS2/",
        "OAS2_RAW_PART1/",
        "OAS2_RAW_PART2/",
        "datasets/raw/",
        "datasets/external/",
    ]

    recommendations = [
        "Use GitHub as the canonical home for backend code and notebooks.",
        "Use Google Drive for raw datasets, extracted dataset folders, checkpoints, and training outputs.",
        "Keep OASIS-1 as the primary 3D branch and Kaggle as the separate 2D comparison branch.",
        "Upload datasets as separate units; do not push the whole workspace with outputs and virtualenv mixed in.",
        "Prefer extracted dataset folders on Drive for training. Zip only for transfer or backup.",
    ]
    if not gitignore_path.exists():
        recommendations.append("Add a workspace-level .gitignore before pushing the codebase to GitHub.")
    if not datasets["oasis1"]["present"]:
        recommendations.append("Confirm the local OASIS-1 path before preparing the Colab training workspace.")
    if not datasets["kaggle_alz"]["present"]:
        recommendations.append("Confirm the local Kaggle Alzheimer dataset path before preparing the 2D comparison branch.")
    if not datasets["oasis2"]["present"]:
        recommendations.append("Keep an empty OASIS-2 slot in Drive, but do not fabricate longitudinal evidence until the dataset exists.")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_statement": (
            "Prepare a clean OASIS-first code/data workflow where GitHub stores backend code, "
            "Google Drive stores datasets and outputs, and Colab runs compute-heavy experiments."
        ),
        "workspace": {
            "workspace_root": str(workspace_root),
            "project_root": str(project_root),
            "gitignore_path": str(gitignore_path),
            "gitignore_present": gitignore_path.exists(),
        },
        "code_for_github": {
            "preferred_repo_root": str(workspace_root),
            "preferred_code_source": "github",
            "keep_local_working_copy": True,
            "include_paths": include_paths,
            "exclude_paths": exclude_paths,
            "notes": [
                "The local machine remains the working copy, but GitHub should be the shareable source for Colab.",
                "Datasets, checkpoints, and outputs should stay out of GitHub.",
            ],
        },
        "datasets_for_drive": datasets,
        "colab_strategy": {
            "preferred_code_source": "github",
            "preferred_dataset_source": "google_drive",
            "preferred_output_target": "google_drive",
            "suggested_drive_layout": {
                "code": "cerebrasense/code/alz_backend",
                "oasis1": "cerebrasense/data/oasis1/raw",
                "kaggle_alz": "cerebrasense/data/kaggle_alz/raw",
                "oasis2": "cerebrasense/data/oasis2/raw",
                "outputs": "cerebrasense/outputs",
            },
            "notes": [
                "Clone or sync code separately from raw data.",
                "Point Colab configs to Drive-mounted dataset folders rather than notebook uploads.",
            ],
        },
        "recommendations": recommendations,
    }


def save_scope_data_plan(
    report: dict[str, Any],
    settings: AppSettings | None = None,
    *,
    file_stem: str = "scope_data_plan",
) -> tuple[Path, Path]:
    """Save the scope data plan as JSON and Markdown."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "readiness")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Scope Data Plan",
        "",
        report["goal_statement"],
        "",
        "## Code For GitHub",
        "",
        f"- preferred_code_source: {report['code_for_github'].get('preferred_code_source')}",
        f"- keep_local_working_copy: {report['code_for_github'].get('keep_local_working_copy')}",
        f"- gitignore_present: {report['workspace'].get('gitignore_present')}",
        "",
        "## Datasets For Drive",
        "",
    ]
    for dataset_name, dataset_info in report.get("datasets_for_drive", {}).items():
        lines.extend(
            [
                f"- {dataset_name}: present={dataset_info.get('present')} drive_root={dataset_info.get('recommended_drive_root')}",
                f"  existing_local_paths={dataset_info.get('existing_local_paths')}",
            ]
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in report.get("recommendations", []))
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
