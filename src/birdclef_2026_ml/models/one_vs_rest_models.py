from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from birdclef_2026_ml.configs import MILConfig
from birdclef_2026_ml.feature_engineering.feature_reweighting import (
    build_feature_scale_vector,
    load_weights,
    apply_profile_weights,
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
class PerClassOneVsRestClassifier:
    """One-vs-rest wrapper that supports per-class sample scopes and scaling."""

    estimators_: list[Any]
    classes_: Array1D
    feature_scales_: Array2D | None = None

    def predict_proba(self, x: Any) -> Array2D:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[np.newaxis, :]
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2D feature matrix")

        positive_probs: list[np.ndarray] = []
        if self.feature_scales_ is None:
            feature_scales = [None] * len(self.estimators_)
        else:
            feature_scales = list(self.feature_scales_)

        for estimator, feature_scale in zip(self.estimators_, feature_scales):
            x_scaled = x_arr if feature_scale is None else x_arr * feature_scale[None, :]
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


def _build_label_scope_by_class_id(
    y_enc: Array1D,
    y_scope: Array1D,
    label_encoder: LabelEncoder,
) -> dict[int, Any]:
    """Build mapping from encoded label id to its unique scope value."""
    label_scope_by_class_id: dict[int, Any] = {}
    for class_id, label in enumerate(label_encoder.classes_):
        scope_values = np.unique(y_scope[y_enc == class_id])
        if len(scope_values) != 1:
            raise ValueError(
                f"Label {label!r} must map to exactly one scope value, got {scope_values.tolist()}"
            )
        label_scope_by_class_id[class_id] = scope_values[0]
    return label_scope_by_class_id


def _fit_per_class_ovr(
    x: Array2D,
    y_enc: Array1D,
    estimator: Any,
    feature_scales: Array2D | None = None,
    alpha: float = 0.5,
    y_scope: Array1D | None = None,
    label_encoder: LabelEncoder | None = None,
) -> PerClassOneVsRestClassifier:
    estimators: list[Any] = []
    n_classes = len(label_encoder.classes_) if label_encoder is not None else int(np.max(y_enc)) + 1
    classes = np.arange(n_classes, dtype=int)

    label_scope_by_class_id: dict[int, Any] | None = None
    if y_scope is not None:
        if label_encoder is None:
            raise ValueError("label_encoder is required when y_scope is provided")
        label_scope_by_class_id = _build_label_scope_by_class_id(
            y_enc=y_enc,
            y_scope=y_scope,
            label_encoder=label_encoder,
        )

    for class_id in classes:
        if y_scope is None:
            train_mask = np.ones(len(y_enc), dtype=bool)
        else:
            train_mask = y_scope == label_scope_by_class_id[class_id]

        y_binary = (y_enc[train_mask] == class_id).astype(int)
        if np.unique(y_binary).size < 2:
            raise ValueError(
                "Each one-vs-rest binary problem must include both positive and negative samples "
                f"after scope filtering; class_id={class_id}"
            )

        x_train = x[train_mask]
        estimator_binary = clone(estimator)
        if feature_scales is not None:
            feature_scale = feature_scales[class_id]
            x_train = apply_profile_weights(x_train, feature_scale[None, :], alpha)
        estimator_binary.fit(x_train, y_binary)
        estimators.append(estimator_binary)

    return PerClassOneVsRestClassifier(
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
        feature_names: list[str],
        class_scope: Any | None = None,
        feature_reweighting: bool = True,
        profile_weights_by_label: dict[str, np.ndarray] | None = None,
        profile_target="primary_label",
        alpha: float = 0.5,  # reweightning parameter
        mil_mode: bool = False,
        mil_config: MILConfig | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
) -> OneVsRestArtifacts:
    """Train one-vs-rest model for a single target vector."""
    _validate_mil_enabled(mil_mode=mil_mode, mil_config=mil_config)
    y = _to_1d(y)
    if class_scope is not None:
        class_scope = _to_1d(class_scope)
        _validate_same_length(x, y, class_scope)
    else:
        _validate_same_length(x, y)

    x_fit = x
    y_fit = y
    class_scope_fit = class_scope
    if mil_mode:
        x_fit, y_fit, bag_sizes = flatten_mil_bags(x=x, y=y)
        if class_scope is not None:
            class_scope_fit = np.repeat(class_scope, bag_sizes)

    label_encoder = _fit_encoder(y)
    y_enc = np.asarray(label_encoder.transform(y_fit), dtype=int)

    x_fit_arr = np.asarray(x_fit, dtype=float)
    if x_fit_arr.ndim != 2:
        raise ValueError("x must be a 2D feature matrix for one-vs-rest training")

    if feature_reweighting:
        if profile_weights_by_label is None:
            profile_weights_by_label = load_weights(profile_target)

        feature_scales = np.vstack([
            build_feature_scale_vector(
                band_weights=profile_weights_by_label.get(
                    str(label),
                    np.ones(x_fit_arr.shape[1], dtype=float),
                ),
                feature_names=feature_names,
            )
            for label in label_encoder.classes_
        ])
        model = _fit_per_class_ovr(
            x=x_fit_arr,
            y_enc=y_enc,
            estimator=estimator,
            feature_scales=feature_scales,
            alpha=alpha,
            y_scope=class_scope_fit,
            label_encoder=label_encoder,
        )
    else:
        model = _fit_per_class_ovr(
            x=x_fit_arr,
            y_enc=y_enc,
            estimator=estimator,
            y_scope=class_scope_fit,
            label_encoder=label_encoder,
        )

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
        primary_label_estimator: Any,
        feature_names: list[str],
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

    class_name_artifacts = train_one_vs_rest_model(
        x=x,
        y=y_class_name,
        estimator=class_name_estimator,
        feature_names=feature_names,
        feature_reweighting=feature_reweighting,
        profile_weights_by_label=profile_weights_by_label,
        profile_target="class_name",
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
        class_scope=y_class_name,
        feature_reweighting=feature_reweighting,
        profile_weights_by_label=profile_weights_by_label,
        profile_target="primary_label",
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
