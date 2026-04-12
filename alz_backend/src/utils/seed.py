"""Random seed helpers for reproducible training and evaluation."""

from __future__ import annotations

import os
import random
from typing import Any


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set Python, NumPy, and torch seeds for reproducible workflows."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def build_seed_snapshot(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Return a JSON-safe seed configuration snapshot."""

    return {
        "seed": seed,
        "deterministic": deterministic,
        "pythonhashseed": str(seed),
    }
