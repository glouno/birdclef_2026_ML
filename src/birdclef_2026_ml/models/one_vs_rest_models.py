from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

from birdclef_2026_ml.feature_engineering.configs import MILConfig
from birdclef_2026_ml.models.hierarchical_models import (
    _fit_encoder,
    _predict_proba_aligned,
    _softmax_with_neg_inf,
    _to_1d,
    _validate_same_length,
)
from birdclef_2026_ml.models.mil_learning import (
    _validate_mil_enabled,
    flatten_mil_bags,
    predict_mil_proba,
)

# Type aliases used throughout this module for readability.
Array1D = np.ndarray
Array2D = np.ndarray


@dataclass
class OneVsRestArtifacts:
    """Trained state for one-vs-rest single-target classification."""

    model: OneVsRestClassifier
    label_encoder: LabelEncoder
    mil_mode: bool = False
    mil_config: MILConfig | None = None


@dataclass
class DualOneVsRestArtifacts:
    """Trained state for paired targets: class_name and primary_label."""

    class_name: OneVsRestArtifacts
    primary_label: OneVsRestArtifacts


def _predict_proba_ovr(artifacts: OneVsRestArtifacts, x: Any) -> Array2D:
    """Return class probabilities aligned to encoder class IDs.

    Preference order: predict_proba -> decision_function (softmax fallback).
    """
    n_classes = len(artifacts.label_encoder.classes_)
    all_class_ids = np.arange(n_classes, dtype=int)

    if hasattr(artifacts.model, "predict_proba"):
        return _predict_proba_aligned(artifacts.model, x, all_class_ids)

    if hasattr(artifacts.model, "decision_function"):
        scores = np.asarray(artifacts.model.decision_function(x), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])

        full_scores = np.full((scores.shape[0], n_classes), -np.inf, dtype=float)
        classes = np.asarray(artifacts.model.classes_, dtype=int)
        full_scores[:, classes] = scores
        return _softmax_with_neg_inf(full_scores)

    raise ValueError(
        "OneVsRestClassifier requires a base estimator exposing "
        "predict_proba or decision_function"
    )


def train_one_vs_rest_model(
        x: Any,
        y: Any,
        estimator: Any,
        mil_mode: bool = False,
        mil_config: MILConfig | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
) -> OneVsRestArtifacts:
    """Train one-vs-rest model for a single target vector."""
    _validate_mil_enabled(mil_mode=mil_mode, mil_config=mil_config)
    y = _to_1d(y)
    _validate_same_length(x, y)

    x_fit = x
    y_fit = y
    if mil_mode:
        x_fit, y_fit, _ = flatten_mil_bags(x=x, y=y)

    label_encoder = _fit_encoder(y)
    y_enc = np.asarray(label_encoder.transform(y), dtype=int)
    if mil_mode:
        y_enc = np.asarray(label_encoder.transform(y_fit), dtype=int)

    model = OneVsRestClassifier(
        estimator=clone(estimator),
        n_jobs=n_jobs,
        verbose=verbose,
    )
    model.fit(x_fit, y_enc)

    return OneVsRestArtifacts(
        model=model,
        label_encoder=label_encoder,
        mil_mode=mil_mode,
        mil_config=mil_config,
    )


def predict_proba_one_vs_rest(artifacts: OneVsRestArtifacts, x: Any) -> Array2D:
    """Predict class probabilities for single-target one-vs-rest."""
    if artifacts.mil_mode:
        if artifacts.mil_config is None:
            raise ValueError("MIL artifacts require mil_config for prediction")
        return predict_mil_proba(
            predict_instance_proba_fn=lambda x_flat: _predict_proba_ovr(artifacts, x_flat),
            x=x,
            mil_cfg=artifacts.mil_config,
        )
    return _predict_proba_ovr(artifacts, x)


def predict_one_vs_rest(artifacts: OneVsRestArtifacts, x: Any) -> Array1D:
    """Predict hard labels for single-target one-vs-rest."""
    proba = predict_proba_one_vs_rest(artifacts, x)
    pred_ids = np.argmax(proba, axis=1)
    return artifacts.label_encoder.inverse_transform(pred_ids)


def train_dual_one_vs_rest_models(
        x: Any,
        y_class_name: Any,
        y_primary_label: Any,
        class_name_estimator: Any,
        primary_label_estimator: Any | None = None,
        mil_mode: bool = False,
        mil_config: MILConfig | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
) -> DualOneVsRestArtifacts:
    """Train paired one-vs-rest models for class_name and primary_label."""
    y_class_name = _to_1d(y_class_name)
    y_primary_label = _to_1d(y_primary_label)
    _validate_same_length(x, y_class_name, y_primary_label)

    if primary_label_estimator is None:
        primary_label_estimator = class_name_estimator

    class_name_artifacts = train_one_vs_rest_model(
        x=x,
        y=y_class_name,
        estimator=class_name_estimator,
        mil_mode=mil_mode,
        mil_config=mil_config,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    primary_label_artifacts = train_one_vs_rest_model(
        x=x,
        y=y_primary_label,
        estimator=primary_label_estimator,
        mil_mode=mil_mode,
        mil_config=mil_config,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return DualOneVsRestArtifacts(
        class_name=class_name_artifacts,
        primary_label=primary_label_artifacts,
    )


def predict_proba_dual_one_vs_rest(
        artifacts: DualOneVsRestArtifacts,
        x: Any,
) -> dict[str, Array2D]:
    """Predict probabilities for both class_name and primary_label."""
    return {
        "class_name": predict_proba_one_vs_rest(artifacts.class_name, x),
        "primary_label": predict_proba_one_vs_rest(artifacts.primary_label, x),
    }


def predict_dual_one_vs_rest(
        artifacts: DualOneVsRestArtifacts,
        x: Any,
) -> dict[str, Array1D]:
    """Predict hard labels for both class_name and primary_label."""
    return {
        "class_name": predict_one_vs_rest(artifacts.class_name, x),
        "primary_label": predict_one_vs_rest(artifacts.primary_label, x),
    }
