from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_multiclass(
    y_true,
    y_pred,
    y_proba,
    classes,
    target_name: str,
    approach: str,
):
    """Evaluate multiclass predictions with requested macro metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=float)
    classes = np.asarray(classes)

    # Keep only labels known by the trained model (robust to rare unseen labels in val).
    known_mask = np.isin(y_true, classes)
    if not np.any(known_mask):
        raise ValueError(f"No known validation labels for target={target_name}.")

    if not np.all(known_mask):
        dropped = int((~known_mask).sum())
        print(f"[{target_name} / {approach}] Dropped {dropped} rows with unseen labels.")

    y_true = y_true[known_mask]
    y_pred = y_pred[known_mask]
    y_proba = y_proba[known_mask]

    if y_proba.shape[1] != len(classes):
        raise ValueError(
            f"Probability column mismatch for {target_name}: "
            f"{y_proba.shape[1]} != {len(classes)}"
        )

    acc = accuracy_score(y_true, y_pred)

    macro_prec, macro_rec, macro_f1, macro_support = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    y_true_bin = label_binarize(y_true, classes=classes)
    try:
        macro_roc_auc = roc_auc_score(
            y_true_bin,
            y_proba,
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        macro_roc_auc = np.nan

    print(f"\n===== {target_name} | {approach} =====")
    print(f"n_samples={len(y_true)}")
    print(f"accuracy={acc:.4f}")
    print(f"macro_precision={macro_prec:.4f}")
    print(f"macro_recall={macro_rec:.4f}")
    print(f"macro_f1={macro_f1:.4f}")
    print(f"macro_roc_auc={macro_roc_auc:.4f}" if np.isfinite(macro_roc_auc) else "macro_roc_auc=nan")

    return {
        "target": target_name,
        "approach": approach,
        "n_samples": len(y_true),
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_roc_auc,
        "macro_support": macro_support,
    }


def plot_confusion_matrix(y_true, y_pred, labels=None, ax=None, normalize="true", cmap="Blues", annot=True, fmt="g"):
    """
    Plot a normalized confusion matrix using seaborn heatmap.
    Args:
        y_true: Ground truth labels (array-like)
        y_pred: Predicted labels (array-like)
        labels: List/array of label names (optional, will use sorted unique labels if None)
        ax: matplotlib axis to plot on (optional)
        normalize: Normalization mode for confusion_matrix (default: "true")
        cmap: Colormap for heatmap
        annot: Annotate cells with values
        fmt: Format for annotations
    Returns:
        The matplotlib axis with the plot.
    """

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Normalized Confusion Matrix")
    return ax
