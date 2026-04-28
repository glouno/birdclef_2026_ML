from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from birdclef_2026_ml.configs import MILConfig
from birdclef_2026_ml.feature_engineering.feature_reweighting import (
    build_feature_scale_vector,
    load_weights,
    apply_profile_weights,
)
from birdclef_2026_ml.models.hierarchical_models import (
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
    scalers_: StandardScaler | dict[Any, StandardScaler]
    class_weights_: Any = None  # dict[scope_value, Array2D]
    estimator_scope_values_: list[Any] | None = None
    feature_scales_: Array2D | None = None
    feature_alpha_: float = 0.5

    def _transform(self, x: np.ndarray, scope_value: Any | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        if self.feature_scales_ is not None:
            feature_scale = self.feature_scales_[0] if scope_value is None else None
            if feature_scale is not None:
                apply_profile_weights(x, feature_scale[None, :], self.feature_alpha_)

        if self.scalers_ is None:
            return x

        if isinstance(self.scalers_, dict):
            scaler = self.scalers_[scope_value]
        else:
            scaler = self.scalers_

        return scaler.transform(x)

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
        if self.estimator_scope_values_ is None:
            estimator_scope_values = [None] * len(self.estimators_)
        else:
            estimator_scope_values = self.estimator_scope_values_

        for estimator, feature_scale, scope_value in zip(
            self.estimators_, feature_scales, estimator_scope_values
        ):
            if feature_scale is None:
                x_scaled = x_arr
            else:
                apply_profile_weights(
                    x_arr,
                    feature_scale[None, :],
                    self.feature_alpha_,
                )
                x_scaled = x_arr
            if self.scalers_ is not None:
                x_scaled = self._transform(x_scaled, scope_value)
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

        # for i, est in enumerate(self.estimators_):
        #     scores = est.decision_function(x_scaled)
        #     print("scores", i, np.min(scores), np.max(scores), np.mean(scores))

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
                f"Label {label!r} ({class_id}) must map to exactly one scope value, got {scope_values.tolist()}"
            )
        label_scope_by_class_id[class_id] = scope_values[0]
    return label_scope_by_class_id


def _validate_binary_targets_for_ovr(
    y_enc: Array1D,
    classes: Array1D,
    y_scope: Array1D | None = None,
    label_scope_by_class_id: dict[int, Any] | None = None,
) -> None:
    for class_id in classes:
        if y_scope is None:
            n_total = len(y_enc)
            n_positive = int(np.count_nonzero(y_enc == class_id))
        else:
            if label_scope_by_class_id is None:
                raise ValueError("label_scope_by_class_id is required when y_scope is provided")
            train_mask = y_scope == label_scope_by_class_id[int(class_id)]
            n_total = int(np.count_nonzero(train_mask))
            n_positive = int(np.count_nonzero(y_enc[train_mask] == class_id))

        n_negative = n_total - n_positive
        if n_positive == 0 or n_negative == 0:
            raise ValueError(
                "Each one-vs-rest binary problem must include both positive and negative samples "
                f"after scope filtering; class_id={int(class_id)}, positives={n_positive}, negatives={n_negative}"
            )


def scaler_transform(x, scalers, scope_value):
    if isinstance(scalers, dict):
        return scalers[scope_value].transform(x)
    return scalers.transform(x)


def _fit_per_class_ovr_incremental(
    x: Array2D,
    y_enc: Array1D,
    *,
    estimator: Any,
    feature_scales: Array2D | None = None,
    alpha: float = 0.5,
    y_scope: Array1D | None = None,
    label_encoder: LabelEncoder,
    batch_size: int,
    epochs: int = 1,
) -> PerClassOneVsRestClassifier:
    n_classes = len(label_encoder.classes_)
    classes = np.arange(n_classes, dtype=int)

    label_scope_by_class_id: dict[int, Any] | None = None
    if y_scope is not None:
        label_scope_by_class_id = _build_label_scope_by_class_id(
            y_enc=y_enc,
            y_scope=y_scope,
            label_encoder=label_encoder,
        )

    _validate_binary_targets_for_ovr(
        y_enc=y_enc,
        classes=classes,
        y_scope=y_scope,
        label_scope_by_class_id=label_scope_by_class_id,
    )

    estimators: list[Any] = []
    binary_classes = np.array([0, 1], dtype=int)
    if y_scope is None:
        scalers: StandardScaler | dict[Any, StandardScaler] = StandardScaler()
        for start in range(0, len(y_enc), batch_size):
            stop = min(start + batch_size, len(y_enc))
            x_batch = np.asarray(x[start:stop], dtype=float)
            if x_batch.shape[0] == 0:
                continue
            scalers.partial_fit(x_batch)
    else:
        scalers = {}
        for scope_value in np.unique(y_scope):
            scaler = StandardScaler()
            scope_idx = np.flatnonzero(y_scope == scope_value)
            for start in range(0, len(scope_idx), batch_size):
                stop = min(start + batch_size, len(scope_idx))
                batch_idx = scope_idx[start:stop]
                x_batch = np.asarray(x[batch_idx], dtype=float)
                if x_batch.shape[0] == 0:
                    continue
                scaler.partial_fit(x_batch)
            scalers[scope_value] = scaler

    estimator_scope_values: list[Any] = []
    for class_id in classes:
        estimator_binary = clone(estimator)
        feature_scale = None if feature_scales is None else feature_scales[int(class_id)]
        scope_value = None
        if label_scope_by_class_id is not None:
            scope_value = label_scope_by_class_id[int(class_id)]
            mask_scope = (y_enc == scope_value)
            idx = np.where(mask_scope)[0]
        else:
            idx = np.arange(len(y_enc))

        idx = resample(idx, replace=True, random_state=42)

        first_batch = True
        for _ in range(epochs):
            for start in range(0, len(idx), batch_size):
                stop = min(start + batch_size, len(y_enc))
                batch_slice = slice(start, stop)
                idx_slice = idx[batch_slice]

                y_batch_enc = y_enc[idx_slice]
                x_batch = np.asarray(x[idx_slice], dtype=float)

                y_binary = (y_batch_enc == class_id).astype(int)
                if feature_scale is not None:
                    apply_profile_weights(x_batch, feature_scale[None, :], 0.5)

                x_batch = scaler_transform(x_batch, scalers, scope_value)

                if first_batch:
                    estimator_binary.partial_fit(x_batch, y_binary, classes=binary_classes)
                    first_batch = False
                else:
                    print("Calling partial fit", y_batch_enc)
                    estimator_binary.partial_fit(x_batch, y_binary)

        if first_batch:
            raise RuntimeError(f"No samples were seen while training class_id={int(class_id)}")
        estimators.append(estimator_binary)
        estimator_scope_values.append(scope_value)

    return PerClassOneVsRestClassifier(
        estimators_=estimators,
        classes_=classes,
        scalers_=scalers,
        estimator_scope_values_=estimator_scope_values,
        feature_scales_=feature_scales,
        feature_alpha_=alpha,
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
        y_enc: Any,
        *,
        label_encoder: LabelEncoder,
        estimator: Any,
        feature_names: list[str],
        class_scope: Any | None = None,
        feature_reweighting: bool = True,
        profile_weights_by_label: dict[str, np.ndarray] | None = None,
        profile_target="primary_label",
        alpha: float = 0.5,  # reweightning parameter
        mil_mode: bool = False,
        mil_config: MILConfig | None = None,
        batch_size: int | None = None,
        epochs: int = 1,
        n_jobs: int | None = None,
        verbose: int = 0,
) -> OneVsRestArtifacts:
    """Train one-vs-rest model for a single target vector."""
    _validate_mil_enabled(mil_mode=mil_mode, mil_config=mil_config)
    y_enc = _to_1d(y_enc)
    if class_scope is not None:
        class_scope = _to_1d(class_scope)
        _validate_same_length(x, y_enc, class_scope)
    else:
        _validate_same_length(x, y_enc)

    x_fit = x
    y_fit = y_enc
    class_scope_fit = class_scope
    if mil_mode:
        x_fit, y_fit, bag_sizes = flatten_mil_bags(x=x, y=y)
        if class_scope is not None:
            class_scope_fit = np.repeat(class_scope, bag_sizes)

    x_fit_arr = np.asarray(x_fit)
    if x_fit_arr.ndim != 2:
        raise ValueError("x must be a 2D feature matrix for one-vs-rest training")

    feature_scales = None
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

    model = _fit_per_class_ovr_incremental(
        x=x_fit_arr,
        y_enc=y_fit,
        estimator=estimator,
        feature_scales=feature_scales,
        alpha=alpha,
        y_scope=class_scope_fit,
        label_encoder=label_encoder,
        batch_size=batch_size,
        epochs=epochs,
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
        label_encoder_class_name: LabelEncoder,
        label_encoder_primary_label: LabelEncoder,
        class_name_estimator: Any,
        primary_label_estimator: Any,
        feature_names: list[str],
        feature_reweighting: bool = True,
        profile_weights_by_label: dict[str, np.ndarray] | None = None,
        mil_mode: bool = False,
        mil_config: MILConfig | None = None,
        batch_size: int | None = None,
        epochs: int = 1,
        n_jobs: int | None = None,
        verbose: int = 0,
) -> DualOneVsRestArtifacts:
    """Train paired one-vs-rest models for class_name and primary_label."""
    y_class_name = _to_1d(y_class_name)
    y_primary_label = _to_1d(y_primary_label)
    _validate_same_length(x, y_class_name, y_primary_label)

    class_name_artifacts = train_one_vs_rest_model(
        x=x,
        y_enc=y_class_name,
        label_encoder=label_encoder_class_name,
        estimator=class_name_estimator,
        feature_names=feature_names,
        feature_reweighting=feature_reweighting,
        profile_weights_by_label=profile_weights_by_label,
        profile_target="class_name",
        mil_mode=mil_mode,
        mil_config=mil_config,
        batch_size=batch_size,
        epochs=epochs,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    primary_label_artifacts = train_one_vs_rest_model(
        x=x,
        y_enc=y_primary_label,
        label_encoder=label_encoder_primary_label,
        estimator=primary_label_estimator,
        feature_names=feature_names,
        class_scope=y_class_name,
        feature_reweighting=feature_reweighting,
        profile_weights_by_label=profile_weights_by_label,
        profile_target="primary_label",
        mil_mode=mil_mode,
        mil_config=mil_config,
        batch_size=batch_size,
        epochs=epochs,
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
