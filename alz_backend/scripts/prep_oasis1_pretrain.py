"""
OASIS-1 Cross-Sectional Pre-training Script.
Designed to be run on Colab before fine-tuning on OASIS-2.
"""
import os
import sys
from pathlib import Path

# Placeholder for Colab integration - run this in a cell
COLAB_CELL = """
import os
os.chdir('/content/cerebrasense/alz_backend')

# Pre-train on OASIS-1 (Cross-sectional)
# This teaches the model the basic anatomy of Dementia
!python scripts/train_oasis2_colab.py \\
    --project-root /content/cerebrasense/alz_backend \\
    --runtime-root /content/drive/MyDrive/Cerebrasensecloud/backend_runtime \\
    --bundle-root /content/drive/MyDrive/Cerebrasensecloud/OASIS-1 \\
    --run-name oasis1_pretrain_base \\
    --epochs 30 \\
    --batch-size 4 \\
    --num-workers 4 \\
    --image-size 96 96 96 \\
    --learning-rate 3e-4 \\
    --scheduler cosine \\
    --weighted-sampling \\
    --device auto
"""

def main():
    print("OASIS-1 Pre-training configuration generated.")
    print("Copy the following command into your Colab notebook for Phase 1:")
    print("-" * 60)
    print(COLAB_CELL)
    print("-" * 60)

if __name__ == "__main__":
    main()
