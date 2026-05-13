import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    label_ranking_average_precision_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize


def recall_at_k(y_true, y_prob, k):
    topk = np.argsort(y_prob, axis=1)[:, -k:]

    hits = 0
    total = 0

    for i in range(y_true.shape[0]):
        true_labels = np.where(y_true[i] == 1)[0]
        total += len(true_labels)
        hits += len(set(topk[i]) & set(true_labels))

    return hits / total


def precision_at_k(y_true, y_prob, k):
    topk = np.argsort(y_prob, axis=1)[:, -k:]

    hits = 0
    total = 0

    for i in range(y_true.shape[0]):
        hits += len(set(topk[i]) & set(np.where(y_true[i] == 1)[0]))
        total += k

    return hits / total


def accuracy_at_k(y_true, y_prob, k):
    topk = np.argsort(y_prob, axis=1)[:, -k:]

    correct = 0

    for i in range(y_true.shape[0]):
        if np.any(y_true[i, topk[i]] == 1):
            correct += 1

    return correct / y_true.shape[0]


def safe_singlelabel_roc_auc(y_true, y_proba, classes):
    y_true_bin = label_binarize(y_true, classes=classes)
    aucs = []
    valid_class_indices = []

    for k in range(y_true_bin.shape[1]):
        y_true_k = y_true_bin[:, k]
        y_proba_k = y_proba[:, k]

        # skip classes with only one label in test
        if np.unique(y_true_k).size < 2:
            continue

        auc = roc_auc_score(y_true_k, y_proba_k)
        aucs.append(auc)
        valid_class_indices.append(k)

    macro_roc_auc = np.mean(aucs) if len(aucs) > 0 else np.nan
    return macro_roc_auc


def safe_multilabel_roc_auc(y_true: np.ndarray, y_score: np.ndarray, average="macro"):
    """
    Computes ROC-AUC for multilabel data safely, skipping classes
    that have only one label present.

    Parameters
    ----------
    y_true : (n_samples, n_classes) binary ground truth
    y_score : (n_samples, n_classes) predicted scores
    average : str
        "macro" or "per_class" or "weighted"

    Returns
    -------
    float
        aggregated ROC-AUC over valid classes
    dict
        per-class AUCs (NaN filtered)
    """

    n_classes = y_true.shape[1]
    aucs = np.full(n_classes, np.nan, dtype=float)

    for c in range(n_classes):
        yt = y_true[:, c]
        ys = y_score[:, c]

        # Skip invalid classes (only one label present)
        if np.unique(yt).size < 2:
            continue

        aucs[c] = roc_auc_score(yt, ys)

    valid = ~np.isnan(aucs)

    if valid.sum() == 0:
        return np.nan, aucs

    if average == "macro":
        return float(np.mean(aucs[valid])), aucs

    if average == "weighted":
        support = y_true.sum(axis=0)
        return float(np.average(aucs[valid], weights=support[valid])), aucs

    if average == "per_class":
        return aucs, aucs

    raise ValueError(f"Unknown average='{average}'")


def safe_singlelabel_average_precision(y_true, y_proba, classes):
    y_true_bin = label_binarize(y_true, classes=classes)

    aps = []

    for k in range(y_true_bin.shape[1]):
        y_true_k = y_true_bin[:, k]
        y_proba_k = y_proba[:, k]

        # Skip classes with only one label present.
        if np.unique(y_true_k).size < 2:
            continue
        ap = average_precision_score(y_true_k, y_proba_k)
        aps.append(ap)

    macro_ap = np.mean(aps) if len(aps) > 0 else np.nan
    return macro_ap


def safe_multilabel_average_precision(y_true: np.ndarray, y_score: np.ndarray, average="macro"):
    """
    Computes average precision for multilabel data safely, skipping classes
    that have only one label present.

    Parameters
    ----------
    y_true : (n_samples, n_classes) binary ground truth
    y_score : (n_samples, n_classes) predicted scores
    average : str
        "macro" or "per_class" or "weighted"

    Returns
    -------
    float
        aggregated average precision over valid classes
    np.ndarray
        per-class APs (NaN filtered)
    """

    n_classes = y_true.shape[1]
    aps = np.full(n_classes, np.nan, dtype=float)

    for c in range(n_classes):
        yt = y_true[:, c]
        ys = y_score[:, c]

        # Skip invalid classes (only one label present)
        if np.unique(yt).size < 2:
            continue

        aps[c] = average_precision_score(yt, ys)

    valid = ~np.isnan(aps)

    if valid.sum() == 0:
        return np.nan, aps

    if average == "macro":
        return float(np.mean(aps[valid])), aps

    if average == "weighted":
        support = y_true.sum(axis=0)
        return float(np.average(aps[valid], weights=support[valid])), aps

    if average == "per_class":
        return aps, aps

    raise ValueError(f"Unknown average='{average}'")


def eval_multiclass_singlelabel(
    y_true,
    y_pred,
    y_proba,
    classes,
):
    """Evaluate multiclass predictions with requested macro metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=float)
    classes = np.asarray(classes)

    # Keep only labels known by the trained model (robust to rare unseen labels in val).
    known_mask = np.isin(y_true, classes)

    if not np.all(known_mask):
        dropped = int((~known_mask).sum())
        print(f"Dropped {dropped} rows with unseen labels.")

    y_true = y_true[known_mask]
    y_pred = y_pred[known_mask]
    y_proba = y_proba[known_mask]
    if y_proba.shape[1] != len(classes):
        raise ValueError(
            f"Probability column mismatch "
            f"{y_proba.shape[1]} != {len(classes)}"
        )
    top5 = top_k_accuracy_score(y_true, y_proba, k=5, labels=classes)

    acc = accuracy_score(y_true, y_pred)
    macro_prec, macro_rec, macro_f1, macro_support = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    macro_roc_auc = safe_singlelabel_roc_auc(y_true, y_proba, classes)
    lrap = label_ranking_average_precision_score(label_binarize(y_true, classes=classes), y_proba)
    macro_ap = safe_singlelabel_average_precision(y_true, y_proba, classes=classes)

    print(f"n_samples={len(y_true)}")
    print(f"accuracy={acc:.4f}")
    print(f"top5_accuracy={top5:.4f}")
    print(f"macro_precision={macro_prec:.4f}")
    print(f"macro_recall={macro_rec:.4f}")
    print(f"macro_f1={macro_f1:.4f}")
    print(f"macro_roc_auc={macro_roc_auc:.4f}" if np.isfinite(macro_roc_auc) else "macro_roc_auc=nan")
    print(f"lrap={lrap:.4f}")
    print(f"mAP={macro_ap:.4f}")

    return {
        "n_samples": len(y_true),
        "accuracy": acc,
        "top5_accuracy": top5,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_roc_auc,
        "macro_support": macro_support,
        "lrap": lrap,
        "mAP": macro_ap,
    }


def plot_confusion_matrix(y_true, y_pred, labels=None, ax=None, cmap="Blues", annot=True, fmt=".2f", plot_labels=False):
    """
    Plot a normalized confusion matrix using seaborn heatmap.
    Args:
        y_true: Ground truth labels (array-like)
        y_pred: Predicted labels (array-like)
        labels: List/array of label names (optional, will use sorted unique labels if None)
        ax: matplotlib axis to plot on (optional)
        cmap: Colormap for heatmap
        annot: Annotate cells with values
        fmt: Format for annotations
    Returns:
        The matplotlib axis with the plot.
    """

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    xticklabels = False
    yticklabels = False
    if plot_labels:
        xticklabels = labels
        yticklabels = labels
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap, ax=ax, vmin=0,
                vmax=1, xticklabels=xticklabels, yticklabels=yticklabels, cbar=False)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Normalized Confusion Matrix")
    return ax
