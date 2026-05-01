from dataclasses import replace
from typing import Any
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.preprocessing import LabelEncoder

from birdclef_2026_ml.models.artifacts import (
    DualOneVsRestArtifacts,
    OneVsRestArtifacts,
    PerClassOneVsRestClassifier,
)
from birdclef_2026_ml.models.one_vs_rest import restrict_by_scope_value
from birdclef_2026_ml.models.weights import (
    compute_pos_neg_sample_weights,
    compute_pos_neg_weights,
    compute_pos_neg_weights_scopes,
)

Array1D = np.ndarray
Array2D = np.ndarray


def _calibrate_binary_estimator(
    estimator: Any,
    x_cal: Array2D,
    y_binary: Array1D,
    *,
    sample_weight: Array1D,
    **calibration_params: Any,
) -> Any:
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(estimator),
        **calibration_params,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*FrozenEstimator does not appear to accept sample_weight.*",
        )
        calibrator.fit(x_cal, y_binary, sample_weight=sample_weight)

    return calibrator


def _calibrate_per_class_ovr_classifier(
    model: PerClassOneVsRestClassifier,
    x_cal: Array2D,
    y_enc: Array1D,
    *,
    label_encoder: LabelEncoder,
    y_scope: Array1D | None = None,
    label_to_scope_mapping: dict[int, int],
    **calibration_params: Any,
) -> PerClassOneVsRestClassifier:
    y_enc = np.asarray(y_enc)
    x_cal = np.asarray(x_cal, dtype=float)
    idx_mapping_scopes = restrict_by_scope_value(np.asarray(y_scope)) if y_scope is not None else None
    pos_neg_weights = compute_pos_neg_weights(y_enc)
    pos_neg_weights_scopes = (
        compute_pos_neg_weights_scopes(y_enc, np.asarray(y_scope))
        if y_scope is not None
        else None
    )

    calibrated_estimators: list[Any] = []
    estimator_scope_values = (
        model.estimator_scope_values_
        if model.estimator_scope_values_ is not None
        else [None] * len(model.estimators_)
    )

    for class_id, estimator, scope_value in zip(model.classes_, model.estimators_, estimator_scope_values):
        if estimator is None:
            calibrated_estimators.append(None)
            continue

        class_id = int(class_id)
        if class_id not in pos_neg_weights:
            calibrated_estimators.append(estimator)
            continue

        x_class = x_cal
        y_class_enc = y_enc
        local_scope_value = scope_value
        local_pos_neg_weights = pos_neg_weights[class_id]

        if idx_mapping_scopes is not None:
            if class_id not in label_to_scope_mapping:
                calibrated_estimators.append(estimator)
                continue
            local_scope_value = label_to_scope_mapping[class_id]
            if local_scope_value not in idx_mapping_scopes:
                calibrated_estimators.append(estimator)
                continue
            if class_id not in pos_neg_weights_scopes[local_scope_value]:
                calibrated_estimators.append(estimator)
                continue
            class_idx = idx_mapping_scopes[local_scope_value]
            x_class = x_cal[class_idx]
            y_class_enc = y_enc[class_idx]
            local_pos_neg_weights = pos_neg_weights_scopes[local_scope_value][class_id]

        y_binary = (y_class_enc == class_id).astype(int)
        if np.unique(y_binary).size < 2:
            calibrated_estimators.append(estimator)
            continue

        x_model = model._transform(x_class, local_scope_value)
        sample_weight = compute_pos_neg_sample_weights(y_binary, local_pos_neg_weights)
        calibrated_estimators.append(
            _calibrate_binary_estimator(
                estimator=estimator,
                x_cal=x_model,
                y_binary=y_binary,
                sample_weight=sample_weight,
                **calibration_params,
            )
        )

    return replace(model, estimators_=calibrated_estimators)


def calibrate_one_vs_rest_artifacts(
    artifacts: Any,
    x_cal: Array2D,
    y_enc: Array1D,
    *,
    y_scope: Array1D | None = None,
    label_to_scope_mapping: dict[int, int],
    **calibration_params: Any,
) -> OneVsRestArtifacts:
    if artifacts.mil_mode:
        raise ValueError("MIL calibration is not supported")

    calibrated_model = _calibrate_per_class_ovr_classifier(
        model=artifacts.model,
        x_cal=x_cal,
        y_enc=y_enc,
        label_encoder=artifacts.label_encoder,
        y_scope=y_scope,
        label_to_scope_mapping=label_to_scope_mapping,
        **calibration_params,
    )
    return OneVsRestArtifacts(
        model=calibrated_model,
        label_encoder=artifacts.label_encoder,
        mil_mode=artifacts.mil_mode,
        mil_config=artifacts.mil_config,
    )


def calibrate_dual_one_vs_rest_artifacts(
    artifacts: Any,
    x_cal: Array2D,
    y_class_name: Array1D,
    y_primary_label: Array1D,
    label_to_scope_mapping: dict[int, int],
    **calibration_params: Any,
) -> DualOneVsRestArtifacts:
    return DualOneVsRestArtifacts(
        class_name=calibrate_one_vs_rest_artifacts(
            artifacts.class_name,
            x_cal=x_cal,
            y_enc=y_class_name,
            label_to_scope_mapping=label_to_scope_mapping,
            **calibration_params,
        ),
        primary_label=calibrate_one_vs_rest_artifacts(
            artifacts.primary_label,
            x_cal=x_cal,
            y_enc=y_primary_label,
            y_scope=y_class_name,
            label_to_scope_mapping=label_to_scope_mapping,
            **calibration_params,
        ),
    )
