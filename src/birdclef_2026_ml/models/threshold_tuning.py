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
from birdclef_2026_ml.models.one_vs_rest_models import predict_proba_one_vs_rest

Array1D = np.ndarray
Array2D = np.ndarray
ScoreFn = Callable[[Array1D, Array1D], float]


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
