"""Generate a lightweight dataset inventory report for the configured source roots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import build_dataset_inventory_snapshot
from src.utils.io_utils import ensure_directory


def main() -> None:
    """Write a dataset inventory snapshot into outputs/reports."""

    report_dir = ensure_directory("outputs/reports")
    report_path = Path(report_dir) / "dataset_inventory.json"
    payload = build_dataset_inventory_snapshot()
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote dataset inventory to {report_path}")


if __name__ == "__main__":
    main()
