from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight

from birdclef_2026_ml.configs import MILConfig

from birdclef_2026_ml.models.hierarchical import (
    _predict_proba_aligned,
    _softmax_with_neg_inf,
    _to_1d,
    _validate_same_length,
)
# from birdclef_2026_ml.models.mil import (
#     _validate_mil_enabled,
#     flatten_mil_bags,
#     predict_mil_proba,
# )

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


# def _build_label_to_scope_mapping(
#     y_enc: Array1D,
#     y_scope: Array1D,
#     y_enc_label_encoder: LabelEncoder,
# ) -> dict[int, Any]:
#     """Build mapping from encoded label id to its unique scope value."""
#     label_to_scope_mapping: dict[int, Any] = {}
#     for class_id, label in enumerate(y_enc_label_encoder.classes_):
#         scope_values = np.unique(y_scope[y_enc == class_id])
#         # label_encoder is fitted on taxonomy, not all species in taxonomy appear in train
#         if len(scope_values) == 0:
#             continue
#         label_to_scope_mapping[class_id] = scope_values[0]
#     return label_to_scope_mapping


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


def _validate_mil_enabled(*, mil_mode: bool, mil_config: MILConfig | None) -> None:
    if mil_mode and mil_config is None:
        raise ValueError("mil_config required when mil_mode=True")


@dataclass(frozen=True)
class _ScopedTrainingData:
    train_idx: Array1D
    train_sample_weight: Array1D
    pos_neg_weights: tuple[float, float]
    scope_value: Any | None
    val_idx: Array1D | None = None


def _fit_scalers(
    x: Array2D,
    idx_mapping_scopes: dict[int, Array1D] | None,
) -> StandardScaler | dict[Any, StandardScaler]:
    if idx_mapping_scopes is None:
        scaler = StandardScaler()
        scaler.fit(x)
        return scaler

    scalers: dict[Any, StandardScaler] = {}
    for scope_value, scope_idx in idx_mapping_scopes.items():
        scaler = StandardScaler()
        scaler.fit(x[scope_idx])
        scalers[scope_value] = scaler
    return scalers


def _build_scoped_training_data(
    class_id: int,
    y_enc: Array1D,
    *,
    sample_weight: Array1D,
    pos_neg_weights,
    sample_weight_scopes: dict[int, Array1D] | None,
    idx_mapping_scopes: dict[int, Array1D] | None,
    pos_neg_weights_scopes,
    label_to_scope_mapping: dict[int, int] | None,
    val_idx_mapping_scopes: dict[int, Array1D] | None,
) -> _ScopedTrainingData:
    if label_to_scope_mapping is None:
        return _ScopedTrainingData(
            train_idx=np.arange(len(y_enc), dtype=np.int64),
            train_sample_weight=sample_weight,
            pos_neg_weights=pos_neg_weights[class_id],
            scope_value=None,
            val_idx=None if val_idx_mapping_scopes is None else np.arange(0, 0, dtype=np.int64),
        )

    scope_value = label_to_scope_mapping[class_id]
    return _ScopedTrainingData(
        train_idx=idx_mapping_scopes[scope_value],
        train_sample_weight=sample_weight_scopes[scope_value],
        pos_neg_weights=pos_neg_weights_scopes[scope_value][class_id],
        scope_value=scope_value,
        val_idx=None if val_idx_mapping_scopes is None else val_idx_mapping_scopes.get(
            scope_value, np.arange(0, 0, dtype=np.int64)
        ),
    )


def _predict_binary_positive_proba(
    estimator: Any,
    x: Array2D,
) -> Array1D:
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(x), dtype=float)
        positive_col = int(np.flatnonzero(np.asarray(estimator.classes_) == 1)[0])
        return proba[:, positive_col]

    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(x), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-scores))

    raise ValueError(
        "Each one-vs-rest estimator must expose predict_proba or decision_function"
    )


def _compute_validation_log_loss(
    estimator: Any,
    x_val: Array2D,
    y_val_binary: Array1D,
    *,
    scalers: StandardScaler | dict[Any, StandardScaler],
    scope_value: Any | None,
) -> float:
    x_val_scaled = scaler_transform(x_val, scalers, scope_value)
    positive_proba = _predict_binary_positive_proba(estimator, x_val_scaled)
    return float(log_loss(y_val_binary, positive_proba, labels=[0, 1]))


def _fit_binary_estimator(
    x: Array2D,
    y_enc: Array1D,
    *,
    class_id: int,
    estimator: Any,
    scalers: StandardScaler | dict[Any, StandardScaler],
    scoped_data: _ScopedTrainingData,
    batch_size: int,
    epochs: int,
    early_stopping: bool,
    n_iter_no_change: int,
    tol: float,
    x_val: Array2D | None,
    y_val: Array1D | None,
) -> Any:
    estimator_binary = clone(estimator)
    first_batch = True
    best_estimator = None
    best_loss = np.inf
    no_improvement_count = 0
    binary_classes = np.array([0, 1], dtype=int)

    for id_epoch in range(epochs):
        epoch_idx = resample(
            scoped_data.train_idx,
            replace=True,
            stratify=None,
            sample_weight=scoped_data.train_sample_weight,
        )
        for start in range(0, len(epoch_idx), batch_size):
            stop = min(start + batch_size, len(epoch_idx))
            idx_slice = epoch_idx[start:stop]
            x_batch = np.asarray(x[idx_slice], dtype=float)
            y_binary = (y_enc[idx_slice] == class_id).astype(int)
            x_batch = scaler_transform(x_batch, scalers, scoped_data.scope_value)
            batch_weight = compute_pos_neg_sample_weights(y_binary, scoped_data.pos_neg_weights)

            if first_batch:
                estimator_binary.partial_fit(
                    x_batch,
                    y_binary,
                    classes=binary_classes,
                    sample_weight=batch_weight,
                )
                first_batch = False
            else:
                estimator_binary.partial_fit(
                    x_batch,
                    y_binary,
                    sample_weight=batch_weight,
                )

        if not early_stopping or x_val is None or y_val is None or scoped_data.val_idx is None:
            continue
        if len(scoped_data.val_idx) == 0:
            continue

        x_val_local = np.asarray(x_val[scoped_data.val_idx], dtype=float)
        y_val_binary = (y_val[scoped_data.val_idx] == class_id).astype(int)
        current_loss = _compute_validation_log_loss(
            estimator_binary,
            x_val_local,
            y_val_binary,
            scalers=scalers,
            scope_value=scoped_data.scope_value,
        )
        if best_loss - current_loss > tol:
            best_loss = current_loss
            best_estimator = deepcopy(estimator_binary)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= n_iter_no_change:
                print(f"Stopped training at {id_epoch} epoch")
                break

    return best_estimator if best_estimator is not None else estimator_binary


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
    label_to_scope_mapping: dict[int, int] | None,
    label_encoder: LabelEncoder,
    batch_size: int,
    epochs: int = 1,
    early_stopping: bool = False,
    n_iter_no_change: int = 5,
    tol: float = 1e-4,
    x_val: Array2D | None = None,
    y_val: Array1D | None = None,
    y_val_scope: Array1D | None = None,
) -> PerClassOneVsRestClassifier:
    classes = np.unique(y_enc)
    class_priors = {
        int(class_id): float(np.mean(y_enc == class_id))
        for class_id in classes
    }

    estimators: list[Any] = []
    fallback_positive_probs: list[float | None] = []
    scalers = _fit_scalers(x, idx_mapping_scopes)
    val_idx_mapping_scopes = None
    if y_val_scope is not None:
        val_idx_mapping_scopes = restrict_by_scope_value(y_val_scope)

    estimator_scope_values: list[Any] = []
    for class_id in classes:
        scoped_data = _build_scoped_training_data(
            int(class_id),
            y_enc,
            sample_weight=sample_weight,
            pos_neg_weights=pos_neg_weights,
            sample_weight_scopes=sample_weight_scopes,
            idx_mapping_scopes=idx_mapping_scopes,
            pos_neg_weights_scopes=pos_neg_weights_scopes,
            label_to_scope_mapping=label_to_scope_mapping,
            val_idx_mapping_scopes=val_idx_mapping_scopes,
        )
        print(f"Training {class_id} with {len(scoped_data.train_idx)} samples")
        scoped_classes = np.unique(y_enc[scoped_data.train_idx])
        if len(scoped_classes) < 2:
            estimators.append(None)
            fallback_positive_probs.append(class_priors[int(class_id)])
            estimator_scope_values.append(scoped_data.scope_value)
            continue

        estimator_binary = _fit_binary_estimator(
            x=x,
            y_enc=y_enc,
            class_id=int(class_id),
            estimator=estimator,
            scalers=scalers,
            scoped_data=scoped_data,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            x_val=x_val,
            y_val=y_val,
        )

        estimators.append(estimator_binary)
        fallback_positive_probs.append(None)
        estimator_scope_values.append(scoped_data.scope_value)

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
        label_to_scope_mapping: dict[int, int] | None,
        mil_mode: bool = False,
        mil_config: MILConfig | None = None,
        batch_size: int,
        epochs: int,
        early_stopping: bool = False,
        n_iter_no_change: int = 5,
        tol: float = 1e-4,
        x_val: Array2D | None = None,
        y_val: Array1D | None = None,
        y_val_scope: Array1D | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
) -> OneVsRestArtifacts:
    """Train one-vs-rest model for a single target vector."""
    _validate_mil_enabled(mil_mode=mil_mode, mil_config=mil_config)
    if early_stopping and (x_val is None or y_val is None):
        raise ValueError("Validation data required when early_stopping=True")

    y_fit = np.asarray(y_enc)
    x_fit = x
    y_val = None if y_val is None else np.asarray(y_val)
    y_val_scope = None if y_val_scope is None else np.asarray(y_val_scope)

    sample_weight = compute_sample_weight(compute_soft_oversampling_weights(y_fit), y_fit)
    sample_weight_scopes = None
    idx_mapping_scopes = None
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
        # label_to_scope_mapping = _build_label_to_scope_mapping(y_fit, y_scope_fit, label_encoder)
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
        early_stopping=early_stopping,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        x_val=x_val,
        y_val=y_val,
        y_val_scope=y_val_scope,
    )

    return OneVsRestArtifacts(
        model=model,
        label_encoder=label_encoder,
        mil_mode=mil_mode,
        mil_config=mil_config,
    )


def predict_proba_one_vs_rest(artifacts: OneVsRestArtifacts, x: Any) -> Array2D:
    """Predict class probabilities for single-target one-vs-rest."""
    # if artifacts.mil_mode:
    #     if artifacts.mil_config is None:
    #         raise ValueError("MIL artifacts require mil_config for prediction")
    #     return predict_mil_proba(
    #         predict_instance_proba_fn=lambda x_flat: _predict_proba_ovr(artifacts, x_flat),
    #         x=x,
    #         mil_cfg=artifacts.mil_config,
    #     )
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
        label_to_scope_mapping: dict[int, int],
        feature_names: list[str],
        mil_mode: bool = False,
        mil_config: MILConfig | None = None,
        batch_size: int,
        epochs: int,
        early_stopping: bool = False,
        n_iter_no_change: int = 5,
        tol: float = 1e-4,
        x_val: Array2D | None = None,
        y_class_name_val: Array1D | None = None,
        y_primary_label_val: Array1D | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
        scope: bool,
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
        label_to_scope_mapping=None,
        mil_mode=mil_mode,
        mil_config=mil_config,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping=early_stopping,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        x_val=x_val,
        y_val=y_class_name_val,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    y_scope = None
    mapping = None
    if scope:
        y_scope = y_class_name
        mapping = label_to_scope_mapping

    primary_label_artifacts = train_one_vs_rest_model(
        x=x,
        y_enc=y_primary_label,
        label_encoder=label_encoder_primary_label,
        estimator=primary_label_estimator,
        feature_names=feature_names,
        y_scope=y_scope,
        label_to_scope_mapping=mapping,
        mil_mode=mil_mode,
        mil_config=mil_config,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping=early_stopping,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        x_val=x_val,
        y_val=y_primary_label_val,
        y_val_scope=y_class_name_val if scope else None,
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
