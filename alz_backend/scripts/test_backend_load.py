import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.dashboard_utils import load_dashboard_data

print("Testing load_dashboard_data...")
data, err = load_dashboard_data(PROJECT_ROOT)
if err:
    print(f"Error: {err}")
else:
    print(f"Success! Loaded {data['summary']['total_subjects']} subjects.")
