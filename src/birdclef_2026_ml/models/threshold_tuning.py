from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from birdclef_2026_ml.models.artifacts import (
    DualOneVsRestArtifacts,
    OneVsRestArtifacts,
    ThresholdTunedDualOneVsRestArtifacts,
    ThresholdTunedOneVsRestArtifacts,
)
from birdclef_2026_ml.models.one_vs_rest import predict_proba_one_vs_rest

Array1D = np.ndarray
Array2D = np.ndarray
ScoreFn = Callable[[Array1D, Array1D], float]


def compute_priors_from_labels(y_class_name, y_primary_label, primary_to_class, n_labels=234):
    y_class_name = np.asarray(y_class_name, dtype=int).reshape(-1)
    y_primary_label = np.asarray(y_primary_label, dtype=int).reshape(-1)
    primary_to_class = np.asarray(primary_to_class, dtype=int)

    if y_primary_label.size and int(y_primary_label.max()) >= n_labels:
        raise ValueError("y_primary_label contains label >= n_labels")
    if primary_to_class.size < n_labels:
        raise ValueError("primary_to_class must have at least n_labels entries")

    n_classes = 0
    if primary_to_class.size:
        n_classes = int(primary_to_class.max()) + 1
    if y_class_name.size:
        n_classes = max(n_classes, int(y_class_name.max()) + 1)

    if y_class_name.size:
        class_counts = np.bincount(y_class_name, minlength=n_classes).astype(float)
        class_priors = class_counts / float(y_class_name.size)
    else:
        class_priors = np.zeros(n_classes, dtype=float)

    if y_primary_label.size:
        primary_counts = np.bincount(y_primary_label, minlength=n_labels).astype(float)
        primary_priors = primary_counts / float(y_primary_label.size)
    else:
        primary_counts = np.zeros(n_labels, dtype=float)
        primary_priors = np.zeros(n_labels, dtype=float)

    missing_mask = primary_counts == 0
    if np.any(missing_mask):
        class_ids = primary_to_class[:n_labels]
        primary_priors[missing_mask] = class_priors[class_ids[missing_mask]]

    return class_priors, primary_priors


def compute_priors_from_multihot(y_class_name, y_primary_label, primary_to_class, n_labels=234):
    y_class_name = np.asarray(y_class_name, dtype=float)
    y_primary_label = np.asarray(y_primary_label, dtype=float)
    primary_to_class = np.asarray(primary_to_class, dtype=int)

    if y_class_name.ndim != 2:
        raise ValueError("y_class_name must be 2D (n_samples, n_classes)")
    if y_primary_label.ndim != 2:
        raise ValueError("y_primary_label must be 2D (n_samples, n_labels)")
    if y_primary_label.shape[1] != n_labels:
        raise ValueError("y_primary_label must have n_labels columns")
    if primary_to_class.size < n_labels:
        raise ValueError("primary_to_class must have at least n_labels entries")

    n_samples = y_class_name.shape[0]
    if n_samples == 0:
        class_priors = np.zeros(y_class_name.shape[1], dtype=float)
    else:
        class_priors = np.mean(y_class_name, axis=0).astype(float)

    if y_primary_label.shape[0] == 0:
        primary_counts = np.zeros(n_labels, dtype=float)
        primary_priors = np.zeros(n_labels, dtype=float)
    else:
        primary_counts = np.sum(y_primary_label, axis=0).astype(float)
        primary_priors = primary_counts / float(y_primary_label.shape[0])

    missing_mask = primary_counts == 0
    if np.any(missing_mask):
        class_ids = primary_to_class[:n_labels]
        primary_priors[missing_mask] = class_priors[class_ids[missing_mask]]

    return class_priors, primary_priors


# def compute_priors(y_class_name, y_primary_label, primary_to_class, n_labels=234):
#     return compute_priors_from_labels(y_class_name, y_primary_label, primary_to_class, n_labels=n_labels)

def _calc_shift(priors, k):
    denom = priors + (1.0 - priors) * k
    return np.divide(priors, denom, out=np.zeros_like(priors), where=denom != 0)


def compute_priors_log_odds_from_labels(
    y_class_name,
    y_primary_label,
    primary_to_class,
    n_classes=234,
    k: float = 1.0,
):

    class_priors, primary_priors = compute_priors_from_labels(
        y_class_name, y_primary_label, primary_to_class, n_labels=n_classes)

    return _calc_shift(class_priors, k), _calc_shift(primary_priors, k)


def compute_priors_log_odds_from_multihot(
    y_class_name,
    y_primary_label,
    primary_to_class,
    n_classes=234,
    k: float = 1.0,
):
    class_priors, primary_priors = compute_priors_from_multihot(
        y_class_name, y_primary_label, primary_to_class, n_labels=n_classes)
    return _calc_shift(class_priors, k), _calc_shift(primary_priors, k)


# def compute_priors_log_odds(
#     y_class_name,
#     y_primary_label,
#     primary_to_class,
#     n_classes=234,
#     k: float = 1.0,
# ):
#     return compute_priors_log_odds_from_labels(
#         y_class_name,
#         y_primary_label,
#         primary_to_class,
#         n_classes=n_classes,
#         k=k,
#     )


def resolve_threshold_score_fn(score: str | ScoreFn) -> tuple[str, ScoreFn]:
    if callable(score):
        score_name = getattr(score, "__name__", "custom_score")
        return score_name, score

    registry: dict[str, ScoreFn] = {
        "accuracy": lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": lambda y_true, y_pred: float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if score not in registry:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported threshold score='{score}'. Supported: {supported}")
    return score, registry[score]


def predict_with_thresholds_from_proba(proba: Array2D, thresholds: Array1D) -> Array1D:
    proba = np.asarray(proba, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if proba.ndim != 2:
        raise ValueError("proba must be 2D")
    if thresholds.shape != (proba.shape[1],):
        raise ValueError("thresholds must have shape (n_classes,)")
    adjusted_scores = proba - thresholds[np.newaxis, :]
    return np.argmax(adjusted_scores, axis=1).astype(int)


def predict_threshold_tuned_one_vs_rest(
    artifacts: ThresholdTunedOneVsRestArtifacts,
    x: Any,
) -> Array1D:
    proba = predict_proba_one_vs_rest(artifacts.base_artifacts, x)
    pred_ids = predict_with_thresholds_from_proba(proba, artifacts.thresholds)
    return artifacts.base_artifacts.label_encoder.inverse_transform(pred_ids)


def _score_thresholds(
    y_true: Array1D,
    proba: Array2D,
    thresholds: Array1D,
    scorer: ScoreFn,
) -> float:
    pred_ids = predict_with_thresholds_from_proba(proba, thresholds)
    return float(scorer(y_true, pred_ids))


def tune_one_vs_rest_thresholds(
    artifacts: OneVsRestArtifacts,
    x: Any,
    y_true: Array1D,
    *,
    score: str | ScoreFn = "macro_f1",
    threshold_grid: Array1D | None = None,
    max_rounds: int = 2,
) -> ThresholdTunedOneVsRestArtifacts:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(predict_proba_one_vs_rest(artifacts, x), dtype=float)
    if proba.ndim != 2:
        raise ValueError("Expected 2D probability matrix")
    if len(y_true) != proba.shape[0]:
        raise ValueError("x and y_true must have same length")

    score_name, scorer = resolve_threshold_score_fn(score)
    grid = np.asarray(
        threshold_grid if threshold_grid is not None else np.linspace(0.05, 0.95, 19),
        dtype=float,
    )
    grid = np.unique(np.clip(np.concatenate([grid, np.array([0.5], dtype=float)]), 0.0, 1.0))

    thresholds = np.full(proba.shape[1], 0.5, dtype=float)
    best_score = _score_thresholds(y_true, proba, thresholds, scorer)

    for _ in range(max_rounds):
        improved = False
        for class_id in range(proba.shape[1]):
            best_threshold = thresholds[class_id]
            for candidate in grid:
                candidate_thresholds = thresholds.copy()
                candidate_thresholds[class_id] = candidate
                candidate_score = _score_thresholds(y_true, proba, candidate_thresholds, scorer)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_threshold = float(candidate)
                    improved = True
            thresholds[class_id] = best_threshold
        if not improved:
            break

    return ThresholdTunedOneVsRestArtifacts(
        base_artifacts=artifacts,
        thresholds=thresholds,
        score_name=score_name,
        best_score=best_score,
    )


def tune_dual_one_vs_rest_thresholds(
    artifacts: DualOneVsRestArtifacts,
    x: Any,
    y_class_name: Array1D,
    y_primary_label: Array1D,
    *,
    score: str | ScoreFn = "macro_f1",
    threshold_grid: Array1D | None = None,
    max_rounds: int = 2,
) -> ThresholdTunedDualOneVsRestArtifacts:
    return ThresholdTunedDualOneVsRestArtifacts(
        class_name=tune_one_vs_rest_thresholds(
            artifacts.class_name,
            x,
            y_class_name,
            score=score,
            threshold_grid=threshold_grid,
            max_rounds=max_rounds,
        ),
        primary_label=tune_one_vs_rest_thresholds(
            artifacts.primary_label,
            x,
            y_primary_label,
            score=score,
            threshold_grid=threshold_grid,
            max_rounds=max_rounds,
        ),
    )
