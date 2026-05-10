from typing import Any

import numpy as np

from birdclef_2026_ml.models.artifacts import OneVsRestArtifacts
from birdclef_2026_ml.paths import load_project_paths

# Type aliases used throughout this module for readability.
Array1D = np.ndarray
Array2D = np.ndarray


def _to_1d(a: Any) -> Array1D:
    """Convert input to 1D ndarray and validate shape."""
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array-like input")
    return arr


def _validate_same_length(*arrays: Any) -> None:
    """Guard against misaligned training inputs."""
    lengths = [len(x) for x in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inputs must have same number of rows, got lengths={lengths}")


def _softmax_with_neg_inf(scores: Array2D) -> Array2D:
    """Softmax that treats -inf as masked classes with zero probability."""
    max_scores = np.max(scores, axis=1, keepdims=True)
    shifted = scores - max_scores
    exp_scores = np.exp(shifted)
    exp_scores[~np.isfinite(scores)] = 0.0
    denom = exp_scores.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return exp_scores / denom


def _predict_proba_aligned(
    model: Any,
    x: Any,
    all_class_ids: Array1D,
    # log: bool = False
) -> Array2D:
    """Align predict_proba output to a global class-id ordering.

    Some estimators only return probabilities for classes seen during fit.
    This helper writes those probabilities into a full matrix with stable columns.
    """
    # if log:
    # proba = np.asarray(model.predict_log_proba(x), dtype=float)
    # else:
    proba = np.asarray(model.predict_proba(x), dtype=float)
    model_classes = np.asarray(model.classes_, dtype=int)

    aligned = np.zeros((proba.shape[0], len(all_class_ids)), dtype=float)
    col_by_class = {int(c): i for i, c in enumerate(all_class_ids)}
    for src_col, cls in enumerate(model_classes):
        aligned[:, col_by_class[int(cls)]] = proba[:, src_col]
    return aligned


def predict_proba_soft_combination(
        artifacts_class_name: OneVsRestArtifacts,
        artifacts_primary_label: OneVsRestArtifacts,
        x: Any,
        y_class_name=None
        # renormalize: bool = False,
) -> Array2D:
    """Soft approach inference.

    Computes:
    P(species|x) <- P(species|x) * P(family_of_species|x)
    """
    if y_class_name is None:
        y_class_name = np.load(load_project_paths().primary_to_class / "primary_to_class.npy")

    n_class_names = len(artifacts_class_name.label_encoder.classes_)
    all_class_names_ids = np.arange(n_class_names, dtype=int)

    n_primary_labels = len(artifacts_primary_label.label_encoder.classes_)
    all_primary_label_ids = np.arange(n_primary_labels, dtype=int)

    # op = np.add if log else np.multiply
    class_name_proba = _predict_proba_aligned(
        artifacts_class_name.model, x, all_class_names_ids,
    )
    primary_label_proba = _predict_proba_aligned(
        artifacts_primary_label.model, x, all_primary_label_ids
    )
    return combine_soft_proba(
        class_name_proba=class_name_proba,
        primary_label_proba=primary_label_proba,
        y_class_name=y_class_name,
    )


def combine_soft_proba(
    class_name_proba: Array2D,
    primary_label_proba: Array2D,
    y_class_name: Array1D | None = None,
    *,
    batch_size: int | None = None,
) -> Array2D:
    """Combine precomputed class_name and primary_label probabilities."""
    class_name_proba = np.asarray(class_name_proba, dtype=float)
    primary_label_proba = np.asarray(primary_label_proba, dtype=float)
    if class_name_proba.ndim != 2 or primary_label_proba.ndim != 2:
        raise ValueError("class_name_proba and primary_label_proba must be 2D")
    if class_name_proba.shape[0] != primary_label_proba.shape[0]:
        raise ValueError("class_name_proba and primary_label_proba must have same row count")

    if y_class_name is None:
        y_class_name = np.load(load_project_paths().primary_to_class / "primary_to_class.npy")
    y_class_name = np.asarray(y_class_name, dtype=int).reshape(-1)

    mn, mx = 1e-12, 1 - 1e-12
    n_rows = class_name_proba.shape[0]
    combined = np.empty_like(primary_label_proba, dtype=float)

    if batch_size is None or n_rows <= batch_size:
        weights = class_name_proba[:, y_class_name]
        combined[:] = np.exp(
            np.log(np.clip(primary_label_proba, mn, mx))
            + np.log(np.clip(weights, mn, mx))
        )
        return combined

    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        weights = class_name_proba[start:stop, y_class_name]
        combined[start:stop] = np.exp(
            np.log(np.clip(primary_label_proba[start:stop], mn, mx))
            + np.log(np.clip(weights, mn, mx))
        )
    return combined


def predict_soft_combination(
    artifacts_class_name: OneVsRestArtifacts,
    artifacts_primary_label: OneVsRestArtifacts,
    x: Any,
    y_class_name,
    # log: bool = False,
) -> Array1D:
    """Return hard class predictions for soft combination approach."""
    proba = predict_proba_soft_combination(
        artifacts_class_name=artifacts_class_name,
        artifacts_primary_label=artifacts_primary_label,
        x=x,
        y_class_name=y_class_name,
        # log=log,
    )
    pred_ids = np.argmax(proba, axis=1)
    return artifacts_primary_label.label_encoder.inverse_transform(pred_ids)
