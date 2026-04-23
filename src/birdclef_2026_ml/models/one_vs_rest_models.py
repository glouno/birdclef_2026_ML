from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

from birdclef_2026_ml.feature_engineering.configs import MILConfig
from birdclef_2026_ml.feature_engineering.feature_reweighting import (
    build_feature_scale_vector,
    load_primary_label_weights,
)
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

    model: Any
    label_encoder: LabelEncoder
    mil_mode: bool = False
    mil_config: MILConfig | None = None


@dataclass
class DualOneVsRestArtifacts:
    """Trained state for paired targets: class_name and primary_label."""

    class_name: OneVsRestArtifacts
    primary_label: OneVsRestArtifacts


@dataclass
class WeightedOneVsRestClassifier:
    """One-vs-rest wrapper that can rescale features independently per class."""

    estimators_: list[Any]
    classes_: Array1D
    feature_scales_: Array2D

    def predict_proba(self, x: Any) -> Array2D:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[np.newaxis, :]
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2D feature matrix")

        positive_probs: list[np.ndarray] = []
        for estimator, feature_scale in zip(self.estimators_, self.feature_scales_):
            x_scaled = x_arr * feature_scale[None, :]
            if hasattr(estimator, "predict_proba"):
                proba = np.asarray(estimator.predict_proba(x_scaled), dtype=float)
                positive_col = int(np.flatnonzero(np.asarray(estimator.classes_) == 1)[0])
                positive_probs.append(proba[:, positive_col])
                continue

            if hasattr(estimator, "decision_function"):
                scores = np.asarray(estimator.decision_function(x_scaled), dtype=float).ravel()
                positive_probs.append(1.0 / (1.0 + np.exp(-scores)))
                continue

            raise ValueError(
                "Each one-vs-rest estimator must expose predict_proba or decision_function"
            )

        proba = np.column_stack(positive_probs)
        denom = proba.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0.0, 1.0, denom)
        return proba / denom


def _fit_weighted_ovr(
    x: Array2D,
    y_enc: Array1D,
    estimator: Any,
    feature_scales: Array2D,
) -> WeightedOneVsRestClassifier:
    estimators: list[Any] = []
    classes = np.arange(feature_scales.shape[0], dtype=int)
    for class_id, feature_scale in enumerate(feature_scales):
        y_binary = (y_enc == class_id).astype(int)
        estimator_binary = clone(estimator)
        estimator_binary.fit(x * feature_scale[None, :], y_binary)
        estimators.append(estimator_binary)

    return WeightedOneVsRestClassifier(
        estimators_=estimators,
        classes_=classes,
        feature_scales_=feature_scales,
    )


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
        feature_names: list[str] | None = None,
        feature_reweighting: bool = True,
        profile_weights_by_label: dict[str, np.ndarray] | None = None,
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

    x_fit_arr = np.asarray(x_fit, dtype=float)
    if x_fit_arr.ndim != 2:
        raise ValueError("x must be a 2D feature matrix for one-vs-rest training")

    inferred_feature_names = feature_names
    if inferred_feature_names is None and hasattr(x, "feature_names"):
        inferred_feature_names = list(getattr(x, "feature_names"))

    if feature_reweighting:
        if profile_weights_by_label is None:
            profile_weights_by_label = load_primary_label_weights()

        feature_scales = np.vstack([
            build_feature_scale_vector(
                band_weights=profile_weights_by_label.get(
                    str(label),
                    np.ones(x_fit_arr.shape[1], dtype=float),
                ),
                n_features=x_fit_arr.shape[1],
                feature_names=inferred_feature_names,
            )
            for label in label_encoder.classes_
        ])
        model = _fit_weighted_ovr(
            x=x_fit_arr,
            y_enc=y_enc,
            estimator=estimator,
            feature_scales=feature_scales,
        )
    else:
        model = OneVsRestClassifier(
            estimator=clone(estimator),
            n_jobs=n_jobs,
            verbose=verbose,
        )
        model.fit(x_fit_arr, y_enc)

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
        feature_names: list[str] | None = None,
        feature_reweighting: bool = True,
        profile_weights_by_label: dict[str, np.ndarray] | None = None,
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
        feature_names=feature_names,
        feature_reweighting=feature_reweighting,
        profile_weights_by_label=profile_weights_by_label,
        mil_mode=mil_mode,
        mil_config=mil_config,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    primary_label_artifacts = train_one_vs_rest_model(
        x=x,
        y=y_primary_label,
        estimator=primary_label_estimator,
        feature_names=feature_names,
        feature_reweighting=feature_reweighting,
        profile_weights_by_label=profile_weights_by_label,
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
