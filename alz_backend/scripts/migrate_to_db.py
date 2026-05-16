"""
Migration script to move CerebraSense research CSVs into the Clinical Database.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db_engine, get_db_session, migrate_csv_to_db

def run_migration():
    print("🚀 Starting Clinical Database Migration...")
    
    engine = get_db_engine("cerebrasense.db")
    db_session = get_db_session(engine)
    
    # Target CSVs
    CSVS = [
        PROJECT_ROOT / "outputs/runs/oasis2/oasis2_colab_improved_v1/evaluation/post_train_test_best_model/predictions.csv",
        PROJECT_ROOT / "outputs/runs/oasis2/oasis2_bias_stability_v1/evaluation/post_train_test_best_model/predictions.csv"
    ]
    
    for csv_path in CSVS:
        if csv_path.exists():
            print(f"📦 Importing {csv_path.name}...")
            migrate_csv_to_db(csv_path, db_session)
        else:
            print(f"⚠️ Skipping {csv_path.name} (not found)")
            
    print("✅ Migration Complete! Database 'cerebrasense.db' is ready.")

if __name__ == "__main__":
    run_migration()
