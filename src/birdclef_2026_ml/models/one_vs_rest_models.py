from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight

from birdclef_2026_ml.configs import MILConfig

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

from birdclef_2026_ml.models.weights import (
    compute_pos_neg_weights,
    compute_pos_neg_sample_weights,
    compute_pos_neg_weights_scopes,
    compute_soft_oversampling_weights
)

from birdclef_2026_ml.models.artifacts import (
    Array1D,
    Array2D,
    OneVsRestArtifacts,
    DualOneVsRestArtifacts,
    PerClassOneVsRestClassifier
)


def _build_label_to_scope_mapping(
    y_enc: Array1D,
    y_scope: Array1D,
    y_enc_label_encoder: LabelEncoder,
) -> dict[int, Any]:
    """Build mapping from encoded label id to its unique scope value."""
    label_to_scope_mapping: dict[int, Any] = {}
    for class_id, label in enumerate(y_enc_label_encoder.classes_):
        scope_values = np.unique(y_scope[y_enc == class_id])
        # label_encoder is fitted on taxonomy, not all species in taxonomy appear in train
        if len(scope_values) == 0:
            continue
        label_to_scope_mapping[class_id] = scope_values[0]
    return label_to_scope_mapping


def restrict_by_scope_value(y_scope: Array1D) -> dict[int, Array1D]:
    mapping = dict()
    y_scope_values = np.unique(y_scope)
    for scope_value in y_scope_values:
        mapping[scope_value] = np.where(y_scope == scope_value)[0]
    return mapping


def scaler_transform(x, scalers, scope_value):
    if isinstance(scalers, dict):
        return scalers[scope_value].transform(x)
    return scalers.transform(x)


def _fit_per_class_ovr_incremental(
    x: Array2D,
    y_enc: Array1D,
    *,
    estimator: Any,
    sample_weight: Array1D,
    pos_neg_weights,
    sample_weight_scopes: dict[int, Array1D] | None,
    idx_mapping_scopes: dict[int, Array1D] | None,
    pos_neg_weights_scopes,
    label_to_scope_mapping,
    label_encoder: LabelEncoder,
    batch_size: int,
    epochs: int = 1,
    # fit_idx: Array1D,
) -> PerClassOneVsRestClassifier:
    classes = np.unique(y_enc)
    class_priors = {
        int(class_id): float(np.mean(y_enc == class_id))
        for class_id in classes
    }

    estimators: list[Any] = []
    fallback_positive_probs: list[float | None] = []
    binary_classes = np.array([0, 1], dtype=int)
    # Fit StandardScalers
    if idx_mapping_scopes is None:
        scalers: StandardScaler | dict[Any, StandardScaler] = StandardScaler()
        scalers.fit(x)
        # for start in range(0, len(y_enc), batch_size):
        #     stop = min(start + batch_size, len(y_enc))
        #     batch_idx = np.arange(start, stop, dtype=np.int64)
        #     x_batch = np.asarray(x[batch_idx], dtype=float)
        #     if x_batch.shape[0] == 0:
        #         continue
        #     scalers.partial_fit(x_batch)
    else:
        scalers = {}
        for scope_value, scope_idx in idx_mapping_scopes.items():
            scaler = StandardScaler()
            # for start in range(0, len(scope_idx), batch_size):
            #     stop = min(start + batch_size, len(scope_idx))
            #     batch_idx = scope_idx[start:stop]
            #     x_batch = np.asarray(x[batch_idx], dtype=float)
            #     if x_batch.shape[0] == 0:
            #         continue
            #     scaler.partial_fit(x_batch)
            scaler.fit(x[scope_idx])
            scalers[scope_value] = scaler

    # Fit OVR models
    estimator_scope_values: list[Any] = []
    for class_id in classes:
        print("Class", class_id)
        sw = sample_weight
        scope_value = None
        pnw = None
        if label_to_scope_mapping is not None:
            scope_value = label_to_scope_mapping[class_id]
            idx = idx_mapping_scopes[scope_value]
            sw = sample_weight_scopes[scope_value]
            pnw = pos_neg_weights_scopes[scope_value][class_id]
        else:
            idx = np.arange(len(y_enc), dtype=np.int64)
            pnw = pos_neg_weights[class_id]

        print("Size of the training dataset:", len(idx))
        scoped_classes = np.unique(y_enc[idx])
        if len(scoped_classes) < 2:
            estimators.append(None)
            fallback_positive_probs.append(class_priors[int(class_id)])
            estimator_scope_values.append(scope_value)
            continue

        estimator_binary = clone(estimator)
        first_batch = True
        idx = resample(idx, replace=True, stratify=None, sample_weight=sw)

        for _ in range(epochs):
            for start in range(0, len(idx), batch_size):
                stop = min(start + batch_size, len(idx))
                batch_slice = slice(start, stop)
                idx_slice = idx[batch_slice]

                y_batch_enc = y_enc[idx_slice]
                x_batch = np.asarray(x[idx_slice], dtype=float)

                y_binary = (y_batch_enc == class_id).astype(int)
                x_batch = scaler_transform(x_batch, scalers, scope_value)
                sw = compute_pos_neg_sample_weights(y_binary, pnw)
                if first_batch:
                    estimator_binary.partial_fit(x_batch, y_binary, classes=binary_classes, sample_weight=sw)
                    first_batch = False
                else:
                    # print("Calling partial fit", np.unique(y_binary, return_counts=True),
                    #       np.unique(y_batch_enc, return_counts=True), pnw)
                    estimator_binary.partial_fit(x_batch, y_binary, sample_weight=sw)

        estimators.append(estimator_binary)
        fallback_positive_probs.append(None)
        estimator_scope_values.append(scope_value)

    return PerClassOneVsRestClassifier(
        estimators_=estimators,
        classes_=classes,
        scalers_=scalers,
        estimator_scope_values_=estimator_scope_values,
        fallback_positive_probs_=fallback_positive_probs,
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
        y_scope: Array1D | None = None,
        mil_mode: bool = False,
        mil_config: MILConfig | None = None,
        batch_size: int,
        epochs: int,
        n_jobs: int | None = None,
        verbose: int = 0,
        # fit_idx: Array1D,
) -> OneVsRestArtifacts:
    """Train one-vs-rest model for a single target vector."""
    _validate_mil_enabled(mil_mode=mil_mode, mil_config=mil_config)

    # x_fit = x
    # y_fit = y_enc
    # class_scope_fit = class_scope

    # TODO: should already be done, no need to call np.repeat.
    # if mil_mode:
    #     x_fit, y_fit, bag_sizes = flatten_mil_bags(x=x, y=y)
    #     if class_scope is not None:
    #         class_scope_fit = np.repeat(class_scope, bag_sizes)

    y_enc = np.asarray(y_enc)
    # fit_idx = np.asarray(fit_idx, dtype=np.int64)
    print(y_enc)
    # y_fit = y_enc[fit_idx]
    y_fit = y_enc
    x_fit = x

    # Compute global weights
    # sample_weight = compute_sample_weight("balanced", y_enc)
    sample_weight = compute_sample_weight(compute_soft_oversampling_weights(y_fit), y_fit)

    # Compute scope related mappings
    sample_weight_scopes = None
    idx_mapping_scopes = None
    label_to_scope_mapping = None
    pos_neg_weights_scopes = None
    pos_neg_weights = compute_pos_neg_weights(y_fit)

    if y_scope is not None:
        y_scope = np.asarray(y_scope)
        # y_scope_fit = y_scope[fit_idx]
        y_scope_fit = y_scope
        idx_mapping_scopes = restrict_by_scope_value(y_scope_fit)
        sample_weight_scopes = {
            scope_value: compute_sample_weight(
                compute_soft_oversampling_weights(y_fit[idx]), y=y_fit[idx]
            )
            for scope_value, idx in idx_mapping_scopes.items()
        }
        label_to_scope_mapping = _build_label_to_scope_mapping(y_fit, y_scope_fit, label_encoder)
        pos_neg_weights_scopes = compute_pos_neg_weights_scopes(y_fit, y_scope_fit)

    model = _fit_per_class_ovr_incremental(
        x=x_fit,
        y_enc=y_fit,
        estimator=estimator,
        sample_weight=sample_weight,
        pos_neg_weights=pos_neg_weights,
        sample_weight_scopes=sample_weight_scopes,
        idx_mapping_scopes=idx_mapping_scopes,
        label_to_scope_mapping=label_to_scope_mapping,
        pos_neg_weights_scopes=pos_neg_weights_scopes,
        label_encoder=label_encoder,
        batch_size=batch_size,
        epochs=epochs,
        # fit_idx=fit_idx,
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
        *,
        label_encoder_class_name: LabelEncoder,
        label_encoder_primary_label: LabelEncoder,
        class_name_estimator: Any,
        primary_label_estimator: Any,
        feature_names: list[str],
        mil_mode: bool = False,
        mil_config: MILConfig | None = None,
        batch_size: int,
        epochs: int,
        n_jobs: int | None = None,
        verbose: int = 0,
        # fit_idx: Array1D,
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
        mil_mode=mil_mode,
        mil_config=mil_config,
        batch_size=batch_size,
        epochs=epochs,
        n_jobs=n_jobs,
        verbose=verbose,
        # fit_idx=fit_idx,
    )
    primary_label_artifacts = train_one_vs_rest_model(
        x=x,
        y_enc=y_primary_label,
        label_encoder=label_encoder_primary_label,
        estimator=primary_label_estimator,
        feature_names=feature_names,
        y_scope=y_class_name,
        mil_mode=mil_mode,
        mil_config=mil_config,
        batch_size=batch_size,
        epochs=epochs,
        n_jobs=n_jobs,
        verbose=verbose,
        # fit_idx=fit_idx,
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
