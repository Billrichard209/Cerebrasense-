"""Publication-friendly plotting helpers for model evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_pyplot() -> object:
    """Lazy-load matplotlib with a headless backend."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_roc_curve_plot(
    *,
    fpr: list[float],
    tpr: list[float],
    auroc: float,
    output_path: Path,
    title: str = "ROC Curve",
    is_defined: bool = True,
    dpi: int = 300,
) -> Path:
    """Save a ROC curve figure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    label = f"AUROC = {auroc:.3f}" if is_defined else "AUROC undefined"
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2.0, label=label)
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_confusion_matrix_plot(
    *,
    confusion_matrix: list[list[int]],
    class_names: tuple[str, str] = ("nondemented", "demented"),
    output_path: Path,
    title: str = "Confusion Matrix",
    dpi: int = 300,
) -> Path:
    """Save a confusion matrix heatmap."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    image = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for row_index, row in enumerate(confusion_matrix):
        for column_index, value in enumerate(row):
            ax.text(column_index, row_index, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_publication_figures(
    *,
    roc_curve: dict[str, Any],
    confusion_matrix: list[list[int]],
    class_names: tuple[str, str],
    output_root: Path,
    title_prefix: str = "OASIS",
) -> dict[str, Path]:
    """Save the standard evaluation figures and return their paths."""

    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "roc_curve": save_roc_curve_plot(
            fpr=list(roc_curve["fpr"]),
            tpr=list(roc_curve["tpr"]),
            auroc=float(roc_curve["auroc"]),
            is_defined=bool(roc_curve["is_defined"]),
            title=f"{title_prefix} ROC Curve",
            output_path=output_root / "roc_curve.png",
        ),
        "confusion_matrix": save_confusion_matrix_plot(
            confusion_matrix=confusion_matrix,
            class_names=class_names,
            title=f"{title_prefix} Confusion Matrix",
            output_path=output_root / "confusion_matrix.png",
        ),
    }
