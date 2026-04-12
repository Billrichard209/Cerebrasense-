"""Build a scope-aware code/data workflow plan for GitHub, Drive, and Colab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scope_data_plan import build_scope_data_plan, save_scope_data_plan  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a practical OASIS-first code/data plan so GitHub, Google Drive, "
            "and Colab use cleanly separated assets."
        )
    )
    parser.add_argument(
        "--file-stem",
        type=str,
        default="scope_data_plan",
        help="Output file stem under outputs/reports/readiness/.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print a summary without writing JSON/Markdown artifacts.",
    )
    return parser


def main() -> None:
    """Build and optionally save the scope data plan."""

    args = build_parser().parse_args()
    report = build_scope_data_plan()
    if not args.no_save:
        json_path, md_path = save_scope_data_plan(report, file_stem=args.file_stem)
        print(f"json_report={json_path}")
        print(f"markdown_report={md_path}")

    print(f"goal_statement={report['goal_statement']}")
    print(f"gitignore_present={report['workspace'].get('gitignore_present')}")
    print(f"oasis1_present={report['datasets_for_drive']['oasis1'].get('present')}")
    print(f"kaggle_present={report['datasets_for_drive']['kaggle_alz'].get('present')}")
    print(f"oasis2_present={report['datasets_for_drive']['oasis2'].get('present')}")
    print(f"recommendations={report.get('recommendations', [])}")


if __name__ == "__main__":
    main()
