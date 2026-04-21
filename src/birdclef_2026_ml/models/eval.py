from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    hamming_loss,
    label_ranking_average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize

from birdclef_2026_ml.feature_engineering import build_feature_vector, build_mil_feature_matrix
from birdclef_2026_ml.feature_engineering.configs import FeatureConfig, MILConfig, PoolingConfig
from birdclef_2026_ml.models.one_vs_rest_models import OneVsRestArtifacts, predict_proba_one_vs_rest
from birdclef_2026_ml.processing.audio_utils import get_path, load_audio


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


def _build_soundscape_segment_input(
    y: np.ndarray,
    artifacts: OneVsRestArtifacts,
    feature_cfg: FeatureConfig,
    pooling_cfg: PoolingConfig,
    mil_cfg: MILConfig | None,
) -> tuple[np.ndarray, list[str]]:
    if artifacts.mil_mode:
        if mil_cfg is None:
            raise ValueError("MIL evaluation requires mil_cfg")
        return build_mil_feature_matrix(
            y=y,
            feature_cfg=feature_cfg,
            pooling_cfg=pooling_cfg,
            mil_cfg=mil_cfg,
        )

    return build_feature_vector(
        y=y,
        feature_cfg=feature_cfg,
        pooling_mode="global",
        pooling_cfg=pooling_cfg,
        chunk_cfg=None,
    )


def evaluate_soundscapes_multilabel(
    soundscapes,
    artifacts: OneVsRestArtifacts,
    feature_cfg: FeatureConfig,
    pooling_cfg: PoolingConfig,
    target_name: str = "primary_label",
    label_col: str = "primary_label_list",
    filename_col: str = "filename",
    start_col: str = "start_sec",
    end_col: str = "end_sec",
    pathroot: str = "train_soundscapes_dir",
    threshold: float = 0.5,
    mil_cfg: MILConfig | None = None,
) -> dict[str, Any]:
    """Evaluate multilabel soundscape predictions from annotated segments.

    Each dataframe row is treated as one annotated segment and evaluated as one
    multilabel sample. No file-level aggregation is performed.
    """
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")

    required_cols = {filename_col, start_col, end_col, label_col}
    missing_cols = required_cols.difference(soundscapes.columns)
    if missing_cols:
        raise ValueError(f"Missing required soundscape columns: {sorted(missing_cols)}")

    mil_cfg = artifacts.mil_config if mil_cfg is None else mil_cfg
    if artifacts.mil_mode and mil_cfg is None:
        raise ValueError("MIL artifacts require mil_cfg for evaluation")

    classes = np.asarray(artifacts.label_encoder.classes_)
    known_classes = set(classes.tolist())

    filenames: list[str] = []
    segment_starts: list[float] = []
    segment_ends: list[float] = []
    y_true_labels: list[list[str]] = []
    y_pred_labels: list[list[str]] = []
    y_score_rows: list[np.ndarray] = []
    dropped_unknown_labels = 0
    skipped_segments = 0
    feature_names_ref: list[str] | None = None

    for row in soundscapes.itertuples(index=False):
        filename = str(getattr(row, filename_col))
        start_seconds = float(getattr(row, start_col))
        end_seconds = float(getattr(row, end_col))
        duration_seconds = end_seconds - start_seconds

        if not np.isfinite(start_seconds) or not np.isfinite(end_seconds) or duration_seconds <= 0.0:
            skipped_segments += 1
            continue

        row_labels: list[str] = []
        for label in getattr(row, label_col):
            label_str = str(label).strip()
            if not label_str:
                continue
            if label_str in known_classes:
                row_labels.append(label_str)
            else:
                dropped_unknown_labels += 1

        audio_path = get_path(pathroot, filename)
        y = load_audio(
            filepath=audio_path,
            sr=feature_cfg.sr,
            offset=start_seconds,
            duration=duration_seconds,
        )
        if np.asarray(y).size == 0:
            skipped_segments += 1
            continue

        segment_input, feature_names = _build_soundscape_segment_input(
            y=np.asarray(y, dtype=float),
            artifacts=artifacts,
            feature_cfg=feature_cfg,
            pooling_cfg=pooling_cfg,
            mil_cfg=mil_cfg,
        )
        if feature_names_ref is None:
            feature_names_ref = list(feature_names)
        elif feature_names != feature_names_ref:
            raise RuntimeError("Inconsistent feature schema across soundscape segments")

        model_input: Any
        if artifacts.mil_mode:
            model_input = [np.asarray(segment_input, dtype=float)]
        else:
            model_input = np.asarray(segment_input, dtype=float).reshape(1, -1)

        segment_proba = np.asarray(predict_proba_one_vs_rest(artifacts, model_input), dtype=float)
        segment_score = segment_proba[0]

        pred_mask = segment_score >= threshold
        pred_labels = classes[pred_mask].tolist()

        filenames.append(filename)
        segment_starts.append(start_seconds)
        segment_ends.append(end_seconds)
        y_true_labels.append(sorted(set(row_labels)))
        y_pred_labels.append(pred_labels)
        y_score_rows.append(segment_score)

    if not y_score_rows:
        raise ValueError("No usable annotated soundscape segments were found for evaluation")

    mlb = MultiLabelBinarizer(classes=classes.tolist())
    mlb.fit([classes.tolist()])
    y_true_bin = mlb.transform(y_true_labels)
    y_pred_bin = mlb.transform(y_pred_labels)
    y_score = np.vstack(y_score_rows)

    subset_accuracy = accuracy_score(y_true_bin, y_pred_bin)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average="macro",
        zero_division=0,
    )
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average="micro",
        zero_division=0,
    )
    samples_prec, samples_rec, samples_f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average="samples",
        zero_division=0,
    )

    try:
        macro_roc_auc = roc_auc_score(y_true_bin, y_score, average="macro")
    except ValueError:
        macro_roc_auc = np.nan

    try:
        micro_roc_auc = roc_auc_score(y_true_bin, y_score, average="micro")
    except ValueError:
        micro_roc_auc = np.nan

    try:
        macro_ap = average_precision_score(y_true_bin, y_score, average="macro")
    except ValueError:
        macro_ap = np.nan

    try:
        micro_ap = average_precision_score(y_true_bin, y_score, average="micro")
    except ValueError:
        micro_ap = np.nan

    try:
        lrap = label_ranking_average_precision_score(y_true_bin, y_score)
    except ValueError:
        lrap = np.nan

    print(f"\n===== {target_name} | soundscape_multilabel_segment =====")
    print(f"n_segments={len(filenames)}")
    print(f"n_classes={len(classes)}")
    print(f"subset_accuracy={subset_accuracy:.4f}")
    print(f"hamming_loss={hamming_loss(y_true_bin, y_pred_bin):.4f}")
    print(f"macro_precision={macro_prec:.4f}")
    print(f"macro_recall={macro_rec:.4f}")
    print(f"macro_f1={macro_f1:.4f}")
    print(f"micro_precision={micro_prec:.4f}")
    print(f"micro_recall={micro_rec:.4f}")
    print(f"micro_f1={micro_f1:.4f}")
    print(f"samples_precision={samples_prec:.4f}")
    print(f"samples_recall={samples_rec:.4f}")
    print(f"samples_f1={samples_f1:.4f}")
    print(f"macro_roc_auc={macro_roc_auc:.4f}" if np.isfinite(macro_roc_auc) else "macro_roc_auc=nan")
    print(f"micro_roc_auc={micro_roc_auc:.4f}" if np.isfinite(micro_roc_auc) else "micro_roc_auc=nan")
    print(f"macro_average_precision={macro_ap:.4f}" if np.isfinite(macro_ap) else "macro_average_precision=nan")
    print(f"micro_average_precision={micro_ap:.4f}" if np.isfinite(micro_ap) else "micro_average_precision=nan")
    print(f"label_ranking_average_precision={lrap:.4f}" if np.isfinite(lrap) else "label_ranking_average_precision=nan")
    if dropped_unknown_labels:
        print(f"dropped_unknown_labels={dropped_unknown_labels}")
    if skipped_segments:
        print(f"skipped_segments={skipped_segments}")

    predictions = pd.DataFrame(
        {
            filename_col: filenames,
            start_col: segment_starts,
            end_col: segment_ends,
            "true_labels": y_true_labels,
            "predicted_labels": y_pred_labels,
        }
    )

    return {
        "target": target_name,
        "approach": "soundscape_multilabel_segment",
        "n_segments": len(filenames),
        "n_classes": len(classes),
        "threshold": threshold,
        "subset_accuracy": subset_accuracy,
        "hamming_loss": hamming_loss(y_true_bin, y_pred_bin),
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "samples_precision": samples_prec,
        "samples_recall": samples_rec,
        "samples_f1": samples_f1,
        "macro_roc_auc": macro_roc_auc,
        "micro_roc_auc": micro_roc_auc,
        "macro_average_precision": macro_ap,
        "micro_average_precision": micro_ap,
        "label_ranking_average_precision": lrap,
        "dropped_unknown_labels": dropped_unknown_labels,
        "skipped_segments": skipped_segments,
        "y_true": y_true_bin,
        "y_pred": y_pred_bin,
        "y_score": y_score,
        "classes": classes,
        "predictions": predictions,
    }
