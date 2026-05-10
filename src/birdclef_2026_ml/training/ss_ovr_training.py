from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from birdclef_2026_ml.configs import (
    artifact_stem,
    build_calibration_config,
    load_calibration_config,
    load_experiment_config,
)
from birdclef_2026_ml.models.artifacts import (
    DualOneVsRestArtifacts,
    PerClassProbabilityCalibrationArtifacts,
    SoundscapeOOFArtifacts,
    SoundscapeOOFFoldArtifacts,
    SoundscapeTargetArtifacts,
)
from birdclef_2026_ml.models.hierarchical import predict_proba_soft_combination
from birdclef_2026_ml.paths import load_project_paths
from birdclef_2026_ml.processing.data_split import iter_soundscapes_oof_splits
from birdclef_2026_ml.processing.memmap_dataset import load_memmap_dataset

Array1D = np.ndarray
Array2D = np.ndarray


def safe_logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _fit_binary_calibrator(
    score: np.ndarray,
    y_binary: np.ndarray,
    *,
    method: str,
):
    score = np.asarray(score, dtype=float)
    y_binary = np.asarray(y_binary, dtype=int)
    # skip degenerate folds
    if np.unique(y_binary).size < 2:
        return None, float(y_binary.mean())

    x = safe_logit(score).reshape(-1, 1)

    if method == "sigmoid":
        calibrator = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
        )
        calibrator.fit(x, y_binary)
        return calibrator, float(y_binary.mean())

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(score, y_binary)
        return calibrator, float(y_binary.mean())
    raise ValueError(f"Unknown method={method}")


def _fit_per_class_probability_calibrator(
    proba: Array2D,
    y_true: Array2D,
    *,
    method: str,
) -> PerClassProbabilityCalibrationArtifacts:
    calibrators: list[Any | None] = []
    constant_probabilities = np.zeros(proba.shape[1], dtype=float)

    for class_id in range(proba.shape[1]):
        calibrator, constant_probability = _fit_binary_calibrator(
            proba[:, class_id],
            y_true[:, class_id],
            method=method,
        )
        calibrators.append(calibrator)
        constant_probabilities[class_id] = constant_probability
    return PerClassProbabilityCalibrationArtifacts(
        method=method,
        calibrators=calibrators,
        constant_probabilities=constant_probabilities,
    )


def apply_per_class_probability_calibrator(
    artifacts: PerClassProbabilityCalibrationArtifacts,
    proba: Array2D,
) -> Array2D:
    calibrated = np.zeros_like(proba, dtype=float)
    for class_id, calibrator in enumerate(artifacts.calibrators):
        if calibrator is None:
            calibrated[:, class_id] = artifacts.constant_probabilities[class_id]
            continue
        if artifacts.method == "sigmoid":
            x = safe_logit(proba[:, class_id]).reshape(-1, 1)
            calibrated[:, class_id] = calibrator.predict_proba(x)[:, 1]
            # calibrated[:, class_id] = calibrator.predict_proba(
            #     np.asarray(proba[:, class_id], dtype=float).reshape(-1, 1)
            # )[:, 1]
            continue
        if artifacts.method == "isotonic":
            calibrated[:, class_id] = calibrator.predict(np.asarray(proba[:, class_id], dtype=float))
            continue
        raise ValueError(f"Unsupported soundscape calibration method='{artifacts.method}'")
    return np.clip(calibrated, 1e-12, 1.0 - 1e-12)


def _predict_multilabel_with_thresholds(proba: Array2D, thresholds: Array1D) -> Array2D:
    return (np.asarray(proba, dtype=float) >= np.asarray(thresholds, dtype=float)[np.newaxis, :]).astype(np.int8)


def _score_multilabel_thresholds(
    y_true: Array2D,
    proba: Array2D,
    thresholds: Array1D,
) -> float:
    y_pred = _predict_multilabel_with_thresholds(proba, thresholds)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def tune_multilabel_thresholds(
    proba: Array2D,
    y_true: Array2D,
    *,
    threshold_grid: Array1D | None = None,
    max_rounds: int = 2,
) -> tuple[Array1D, float]:
    grid = np.asarray(
        threshold_grid if threshold_grid is not None else np.linspace(0.05, 0.95, 19),
        dtype=float,
    )
    grid = np.unique(np.clip(np.concatenate([grid, np.array([0.5], dtype=float)]), 0.0, 1.0))
    thresholds = np.full(proba.shape[1], 0.5, dtype=float)
    best_score = _score_multilabel_thresholds(y_true, proba, thresholds)

    for _ in range(max_rounds):
        improved = False
        for class_id in range(proba.shape[1]):
            best_threshold = thresholds[class_id]
            for candidate in grid:
                candidate_thresholds = thresholds.copy()
                candidate_thresholds[class_id] = candidate
                candidate_score = _score_multilabel_thresholds(y_true, proba, candidate_thresholds)
                # print(f"Candidate {candidate}, score {candidate_score}")
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_threshold = float(candidate)
                    improved = True
            thresholds[class_id] = best_threshold
        if not improved:
            break

    return thresholds, best_score


def _extract_primary_label_targets(y: np.ndarray, n_primary_labels: int) -> Array2D:
    if y.ndim != 3 or y.shape[1] != 2:
        raise ValueError("Expected soundscape targets with shape (n_samples, 2, n_classes)")
    return np.asarray(y[:, 1, :n_primary_labels], dtype=np.int8)


def _save_outputs(
    output_dir: Path,
    *,
    combined_proba: Array2D,
    calibrated_proba: Array2D,
    thresholds: Array1D,
    predictions: Array2D,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "primary_label_combined_proba.npy", combined_proba)
    np.save(output_dir / "primary_label_calibrated_proba.npy", calibrated_proba)
    np.save(output_dir / "primary_label_thresholds.npy", thresholds)
    np.save(output_dir / "primary_label_predictions.npy", predictions)


def run_soundscape_oof_evaluation_and_training(
    run_name: str,
    experiment_name: str,
    *,
    calibration_name: str | None = None,
    reduced: bool = True,
    model_filename: str | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    threshold_grid: Array1D | None = None,
    max_rounds: int = 2,
) -> Path:
    paths = load_project_paths()
    experiment_cfg, config_path = load_experiment_config(run_name, experiment_name)
    experiment_dir = paths.experiment_dir(run_name, experiment_name)
    soundscape_dir = experiment_dir / "soundscapes"
    soundscape_dir.mkdir(parents=True, exist_ok=True)

    stem = artifact_stem(experiment_cfg)
    model_path = experiment_dir / (model_filename or f"{stem}_ovr.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Saved OVR artifacts not found: {model_path}")

    calibration_cfg = (
        load_calibration_config(run_name, calibration_name)[0]
        if calibration_name is not None
        else build_calibration_config()
    )
    calibration_method = str(calibration_cfg.params.get("method", "sigmoid"))

    dataset = load_memmap_dataset(run_name, reduced=reduced, soundscape=True)
    artifacts = joblib.load(model_path)
    if not isinstance(artifacts, DualOneVsRestArtifacts):
        raise TypeError("Soundscape OOF requires DualOneVsRestArtifacts")

    primary_to_class = np.asarray(np.load(paths.primary_to_class / "primary_to_class.npy"), dtype=int)
    combined_proba = np.asarray(
        predict_proba_soft_combination(
            artifacts.class_name,
            artifacts.primary_label,
            dataset.X,
            y_class_name=primary_to_class,
        ),
        dtype=float,
    )
    y_primary_label = _extract_primary_label_targets(
        np.asarray(dataset.y),
        n_primary_labels=combined_proba.shape[1],
    )
    splits = iter_soundscapes_oof_splits(
        y=y_primary_label,
        filenames=np.asarray(dataset.filenames),
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_proba = np.zeros_like(combined_proba, dtype=float)
    fold_artifacts: list[SoundscapeOOFFoldArtifacts] = []

    for fold_id, (train_idx, val_idx) in enumerate(splits):
        calibration = _fit_per_class_probability_calibrator(
            combined_proba[train_idx],
            y_primary_label[train_idx],
            method=calibration_method,
        )

        oof_proba[val_idx] = apply_per_class_probability_calibrator(
            calibration,
            combined_proba[val_idx],
        )
        # oof_proba[val_idx] = combined_proba[val_idx]
        fold_artifacts.append(
            SoundscapeOOFFoldArtifacts(
                fold_id=fold_id,
                train_indices=np.asarray(train_idx, dtype=np.int32),
                val_indices=np.asarray(val_idx, dtype=np.int32),
            )
        )

    thresholds, oof_score = tune_multilabel_thresholds(
        oof_proba,
        y_primary_label,
        threshold_grid=threshold_grid,
        max_rounds=max_rounds,
    )

    full_calibration = _fit_per_class_probability_calibrator(
        combined_proba,
        y_primary_label,
        method=calibration_method,
    )
    final_proba = apply_per_class_probability_calibrator(full_calibration, combined_proba)
    # final_proba = oof_proba
    oof_predictions = _predict_multilabel_with_thresholds(oof_proba, thresholds)
    final_predictions = _predict_multilabel_with_thresholds(final_proba, thresholds)

    soundscape_artifacts = SoundscapeOOFArtifacts(
        source_model_path=str(model_path.resolve()),
        calibration_config=calibration_cfg.to_dict(),
        fold_artifacts=fold_artifacts,
        primary_label=SoundscapeTargetArtifacts(
            calibration=None,
            thresholds=thresholds,
            score_name="macro_f1",
            best_score=oof_score,
        ),
    )

    np.save(soundscape_dir / "fold_count.npy", np.array([len(fold_artifacts)], dtype=np.int32))
    for fold in fold_artifacts:
        np.save(soundscape_dir / f"fold_{fold.fold_id}_train_indices.npy", fold.train_indices)
        np.save(soundscape_dir / f"fold_{fold.fold_id}_val_indices.npy", fold.val_indices)

    _save_outputs(
        soundscape_dir / "oof",
        combined_proba=combined_proba,
        calibrated_proba=oof_proba,
        thresholds=thresholds,
        predictions=oof_predictions,
    )
    _save_outputs(
        soundscape_dir / "final",
        combined_proba=combined_proba,
        calibrated_proba=final_proba,
        thresholds=thresholds,
        predictions=final_predictions,
    )

    joblib.dump(soundscape_artifacts, soundscape_dir / f"{Path(model_path).stem}_soundscape_oof.joblib")
    with open(soundscape_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment_cfg.to_dict(), handle, sort_keys=False)
    with open(soundscape_dir / "config_source.txt", "w", encoding="utf-8") as handle:
        handle.write(str(config_path.resolve()))
    if calibration_name is not None:
        with open(soundscape_dir / "calibration_name.txt", "w", encoding="utf-8") as handle:
            handle.write(calibration_name)

    summary = {
        "primary_label": {
            "score_name": "macro_f1",
            "oof_macro_f1": _score_multilabel_thresholds(y_primary_label, oof_proba, thresholds),
            "final_macro_f1": _score_multilabel_thresholds(y_primary_label, final_proba, thresholds),
        },
    }
    with open(soundscape_dir / "metrics.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)

    return soundscape_dir
