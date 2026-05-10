import joblib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    hamming_loss,
    label_ranking_average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score
)
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize

from birdclef_2026_ml.feature_engineering import build_feature_vector, build_mil_feature_matrix
from birdclef_2026_ml.configs import PipelineConfig
from birdclef_2026_ml.models.one_vs_rest import OneVsRestArtifacts, predict_proba_one_vs_rest
from birdclef_2026_ml.paths import load_project_paths
from birdclef_2026_ml.processing.audio_utils import get_path, load_audio
from birdclef_2026_ml.processing.memmap_dataset import load_memmap_dataset
from birdclef_2026_ml.inference.ovr_inference import predict_from_artifacts_batched
from birdclef_2026_ml.models.hierarchical import predict_proba_soft_combination


def safe_singlelabel_roc_auc(y_true, y_proba, classes):
    y_true_bin = label_binarize(y_true, classes=classes)
    print(y_true_bin)
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

    top2 = top_k_accuracy_score(y_true, y_pred, k=2)
    top3 = top_k_accuracy_score(y_true, y_pred, k=3)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    macro_prec, macro_rec, macro_f1, macro_support = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    macro_roc_auc = safe_singlelabel_roc_auc(y_true, y_proba, classes)

    print(f"\n===== {target_name} | {approach} =====")
    print(f"n_samples={len(y_true)}")
    print(f"accuracy={acc:.4f}")
    print(f"balanced_accuracy={bal_acc:.4f}")
    print(f"top2_accuracy={top2:.4f}")
    print(f"top3_accuracy={top3:.4f}")
    print(f"macro_precision={macro_prec:.4f}")
    print(f"macro_recall={macro_rec:.4f}")
    print(f"macro_f1={macro_f1:.4f}")
    print(f"macro_roc_auc={macro_roc_auc:.4f}" if np.isfinite(macro_roc_auc) else "macro_roc_auc=nan")

    return {
        "target": target_name,
        "approach": approach,
        "n_samples": len(y_true),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "top2_accuracy": top2,
        "top3_accuracy": top3,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_roc_auc,
        "macro_support": macro_support,
    }


def plot_confusion_matrix(y_true, y_pred, labels=None, ax=None, cmap="Blues", annot=True, fmt=".2f"):
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
    if len(labels) < 10:
        xticklabels = labels
        yticklabels = labels
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap, ax=ax, vmin=0,
                vmax=1, xticklabels=xticklabels, yticklabels=yticklabels, cbar=False)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Normalized Confusion Matrix")
    return ax


def eval_train_test(run_name: str, experiment_name: str):
    paths = load_project_paths()
    primary_to_class = np.load(paths.primary_to_class / "primary_to_class.npy")

    experiment_path = paths.experiment_dir(run_name, experiment_name)

    dataset = load_memmap_dataset(run_name=run_name, reduced=True, soundscape=False)

    artifacts = joblib.load(experiment_path / "sgdclassifier_ovr.joblib")
    val_idx = np.load(experiment_path / "val_indices.npy")

    true_class_name = dataset.y[val_idx, 0]
    true_primary_label = dataset.y[val_idx, 1]
    n_class_name = len(artifacts.class_name.label_encoder.classes_)
    n_primary_label = len(artifacts.primary_label.label_encoder.classes_)

    X_val = dataset.X[val_idx]
    probas, preds = predict_from_artifacts_batched(
        X_val, artifacts, batch_size=65536
    )

    def _print_metrics(label: str, roc_auc: float, avg_precision: float):
        roc_str = f"{roc_auc:.4f}" if np.isfinite(roc_auc) else "nan"
        ap_str = f"{avg_precision:.4f}" if np.isfinite(avg_precision) else "nan"
        print(f"{label}: roc_auc={roc_str} avg_precision={ap_str}")

    print("1. Without soft combination")
    class_classes = np.arange(n_class_name)
    primary_classes = np.arange(n_primary_label)
    _print_metrics(
        "class_name",
        safe_singlelabel_roc_auc(true_class_name, probas["class_name"], class_classes),
        safe_singlelabel_average_precision(true_class_name, probas["class_name"], class_classes),
    )
    _print_metrics(
        "primary_label",
        safe_singlelabel_roc_auc(true_primary_label, probas["primary_label"], primary_classes),
        safe_singlelabel_average_precision(true_primary_label, probas["primary_label"], primary_classes),
    )

    print("2. With soft combination")
    probas_soft = predict_proba_soft_combination(
        artifacts_class_name=artifacts.class_name,
        artifacts_primary_label=artifacts.primary_label,
        x=X_val,
        y_class_name=primary_to_class
    )

    _print_metrics(
        "primary_label_soft",
        safe_singlelabel_roc_auc(true_primary_label, probas_soft, primary_classes),
        safe_singlelabel_average_precision(true_primary_label, probas_soft, primary_classes),
    )
    return probas, probas_soft
    # plot_confusion_matrix(true_class_name, preds["class_name"], normalize="true", annot=True)


def eval_soundscapes(run_name: str, experiment_name: str, probas_file: str = None):
    paths = load_project_paths()
    primary_to_class = np.load(paths.primary_to_class / "primary_to_class.npy")

    experiment_path = paths.experiment_dir(run_name, experiment_name)
    soundscape_dir = experiment_path / "soundscapes"
    probas_primary_label = None
    probas_class_name = None
    probas_soft = None
    dataset = load_memmap_dataset(run_name=run_name, reduced=True, soundscape=True)
    artifacts = joblib.load(experiment_path / "sgdclassifier_ovr.joblib")

    if probas_file:
        probas_primary_label = np.load(soundscape_dir / probas_file)

    else:
        probas, _ = predict_from_artifacts_batched(
            dataset.X, artifacts, batch_size=65536
        )
        probas_class_name = probas["class_name"]
        probas_primary_label = probas["primary_label"]

    true_class_name = dataset.y[:, 0, :5]
    true_primary_label = dataset.y[:, 1, :]
    # n_primary_label = len(artifacts.primary_label.label_encoder.classes_)

    def _print_metrics(label: str, roc_auc: float, avg_precision: float):
        roc_str = f"{roc_auc:.4f}" if np.isfinite(roc_auc) else "nan"
        ap_str = f"{avg_precision:.4f}" if np.isfinite(avg_precision) else "nan"
        print(f"{label}: roc_auc={roc_str} avg_precision={ap_str}")

    print("1. Without soft combination")
    if probas_class_name is not None:
        roc_auc, _ = safe_multilabel_roc_auc(true_class_name, probas_class_name)
        avg_precision, _ = safe_multilabel_average_precision(true_class_name, probas_class_name)
        _print_metrics("class_name", roc_auc, avg_precision)
        lrap = label_ranking_average_precision_score(true_class_name, probas_class_name)
        print(f"class_name: lrap={lrap:.4f}" if np.isfinite(lrap) else "class_name: lrap=nan")

    roc_auc, _ = safe_multilabel_roc_auc(true_primary_label, probas_primary_label)
    avg_precision, _ = safe_multilabel_average_precision(true_primary_label, probas_primary_label)
    _print_metrics("primary_label", roc_auc, avg_precision)
    lrap = label_ranking_average_precision_score(true_primary_label, probas_primary_label)
    print(f"primary_label: lrap={lrap:.4f}" if np.isfinite(lrap) else "primary_label: lrap=nan")

    if probas_class_name is not None:
        print("2. With soft combination")
        probas_soft = predict_proba_soft_combination(
            artifacts_class_name=artifacts.class_name,
            artifacts_primary_label=artifacts.primary_label,
            x=dataset.X,
            y_class_name=primary_to_class
        )

        roc_auc, _ = safe_multilabel_roc_auc(true_primary_label, probas_soft)
        avg_precision, _ = safe_multilabel_average_precision(true_primary_label, probas_soft)
        _print_metrics("primary_label_soft", roc_auc, avg_precision)
        lrap = label_ranking_average_precision_score(true_primary_label, probas_soft)
        print(f"primary_label_soft: lrap={lrap:.4f}" if np.isfinite(lrap) else "primary_label_soft: lrap=nan")
    return probas_class_name, probas_primary_label, probas_soft
    # plot_confusion_matrix(true_class_name, preds["class_name"], normalize="true", annot=True)
