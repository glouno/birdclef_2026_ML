from dataclasses import dataclass
from typing import Any
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

from birdclef_2026_ml.configs import MILConfig

Array1D = np.ndarray
Array2D = np.ndarray


@dataclass
class MILFeatureBags:
    bags: list[Array2D]
    feature_names: list[str]
    bag_ids: Array1D | None = None

    def __len__(self) -> int:
        return len(self.bags)

    def __iter__(self):
        return iter(self.bags)


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
class ThresholdTunedOneVsRestArtifacts:
    """One-vs-rest artifacts with tuned per-class decision thresholds."""

    base_artifacts: OneVsRestArtifacts
    thresholds: Array1D
    score_name: str
    best_score: float


@dataclass
class ThresholdTunedDualOneVsRestArtifacts:
    """Dual one-vs-rest artifacts with tuned per-target thresholds."""

    class_name: ThresholdTunedOneVsRestArtifacts
    primary_label: ThresholdTunedOneVsRestArtifacts


@dataclass
class PerClassProbabilityCalibrationArtifacts:
    """Per-class probability calibration state for multilabel soundscape outputs."""

    method: str
    calibrators: list[Any | None]
    constant_probabilities: Array1D


@dataclass
class SoundscapeTargetArtifacts:
    """Calibration + threshold state for one multilabel target."""

    calibration: PerClassProbabilityCalibrationArtifacts
    thresholds: Array1D
    score_name: str
    best_score: float


@dataclass
class SoundscapeOOFFoldArtifacts:
    """Stored train/validation indices for one OOF fold."""

    fold_id: int
    train_indices: Array1D
    val_indices: Array1D


@dataclass
class SoundscapeOOFArtifacts:
    """Full soundscape post-processing state fitted on OOF predictions."""

    source_model_path: str
    calibration_config: dict[str, Any]
    fold_artifacts: list[SoundscapeOOFFoldArtifacts]
    primary_label: SoundscapeTargetArtifacts


@dataclass
class SecondStagePrimaryLabelArtifacts:
    """Stage-2 primary-label OVR model trained on bagged stage-1 probabilities."""

    model: Any
    label_encoder: LabelEncoder
    active_label_indices: Array1D
    feature_names: Array1D
    source_model_path: str
    training_summary: dict[str, Any]


@dataclass
class MILSecondStageCleanAudioArtifacts:
    """Stage-2 per-class classifiers for clean-audio MIL bag features."""

    estimators: list[Any | None]
    label_encoder: LabelEncoder
    fallback_positive_probs: list[float | None]
    feature_names: Array1D | None = None
    training_summary: dict[str, Any] | None = None


@dataclass
class PerClassOneVsRestClassifier:
    """One-vs-rest wrapper that supports per-class sample scopes and scaling."""

    estimators_: list[Any]
    classes_: Array1D
    scalers_: StandardScaler | dict[Any, StandardScaler]
    class_weights_: Any = None  # dict[scope_value, Array2D]
    estimator_scope_values_: list[Any] | None = None
    fallback_positive_probs_: list[float | None] | None = None
    feature_alpha_: float = 0.5

    def _transform(self, x: np.ndarray, scope_value: Any | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
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

        if self.estimator_scope_values_ is None:
            estimator_scope_values = [None] * len(self.estimators_)
        else:
            estimator_scope_values = self.estimator_scope_values_

        if self.fallback_positive_probs_ is None:
            fallback_positive_probs = [None] * len(self.estimators_)
        else:
            fallback_positive_probs = self.fallback_positive_probs_

        for estimator, scope_value, fallback_positive_prob in zip(
            self.estimators_, estimator_scope_values, fallback_positive_probs
        ):
            if estimator is None:
                if fallback_positive_prob is None:
                    raise ValueError("Missing fallback probability for untrained estimator")
                positive_probs.append(np.full(x_arr.shape[0], fallback_positive_prob, dtype=float))
                continue

            x_local = x_arr
            if self.scalers_ is not None:
                x_local = self._transform(x_arr, scope_value)
            if hasattr(estimator, "predict_proba"):
                proba = np.asarray(estimator.predict_proba(x_local), dtype=float)
                positive_col = int(np.flatnonzero(np.asarray(estimator.classes_) == 1)[0])
                positive_probs.append(proba[:, positive_col])
                continue

            if hasattr(estimator, "decision_function"):
                scores = np.asarray(estimator.decision_function(x_local), dtype=float).ravel()
                positive_probs.append(1.0 / (1.0 + np.exp(-scores)))
                continue

            raise ValueError(
                "Each one-vs-rest estimator must expose predict_proba or decision_function"
            )
        proba = np.column_stack(positive_probs)
        return proba

    def predict_log_proba(self, x: Any):
        proba = self.predict_proba(x)
        proba = np.clip(proba, 1e-12, 1.0 - 1e-12)
        return np.log(proba)

    def decision_function(self, x: Any) -> Array2D:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[np.newaxis, :]
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2D feature matrix")

        positive_scores: list[np.ndarray] = []

        if self.estimator_scope_values_ is None:
            estimator_scope_values = [None] * len(self.estimators_)
        else:
            estimator_scope_values = self.estimator_scope_values_

        if self.fallback_positive_probs_ is None:
            fallback_positive_probs = [None] * len(self.estimators_)
        else:
            fallback_positive_probs = self.fallback_positive_probs_

        for estimator, scope_value, fallback_positive_prob in zip(
            self.estimators_, estimator_scope_values, fallback_positive_probs
        ):
            if estimator is None:
                if fallback_positive_prob is None:
                    raise ValueError("Missing fallback probability for untrained estimator")
                clipped_prob = np.clip(fallback_positive_prob, 1e-12, 1.0 - 1e-12)
                logit = np.log(clipped_prob / (1.0 - clipped_prob))
                positive_scores.append(np.full(x_arr.shape[0], logit, dtype=float))
                continue

            x_local = x_arr
            if self.scalers_ is not None:
                x_local = self._transform(x_arr, scope_value)

            scores = np.asarray(estimator.decision_function(x_local), dtype=float).ravel()
            positive_scores.append(scores)

        scores = np.column_stack(positive_scores)
        return scores
