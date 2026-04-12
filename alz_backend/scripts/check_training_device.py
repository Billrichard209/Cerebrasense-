"""Profile local device readiness for OASIS 3D MONAI training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.device_profile import collect_training_device_profile, save_training_device_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Check whether this laptop is ready for 3D MONAI training.")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the device profile check."""

    args = parse_args()
    profile = collect_training_device_profile()
    output_path = None if args.no_save else save_training_device_profile(profile, args.output_path)

    print(json.dumps(profile.to_dict(), indent=2))
    if output_path is not None:
        print(f"saved_profile={output_path}")


if __name__ == "__main__":
    main()
