from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.linear_model import SGDClassifier

from birdclef_2026_ml.configs import (
    CalibrationConfig,
    ExperimentConfig,
    load_experiment_config,
    load_calibration_config,
    artifact_stem,
    build_calibration_params,
    build_estimator
)
from birdclef_2026_ml.models.calibration import calibrate_dual_one_vs_rest_artifacts
from birdclef_2026_ml.models.artifacts import (
    DualOneVsRestArtifacts,
    OneVsRestArtifacts,
)
from birdclef_2026_ml.models.one_vs_rest import (
    train_dual_one_vs_rest_models,
    predict_proba_dual_one_vs_rest,
    predict_dual_one_vs_rest,
    predict_proba_one_vs_rest,
)
from birdclef_2026_ml.models.threshold_tuning import (
    predict_threshold_tuned_one_vs_rest,
    tune_dual_one_vs_rest_thresholds,
    tune_one_vs_rest_thresholds,
)
from birdclef_2026_ml.paths import load_project_paths
from birdclef_2026_ml.processing.data_split import get_idx_from_filenames
from birdclef_2026_ml.processing.memmap_dataset import load_memmap_dataset


def train_and_save_ovr_models_chunks(run_name: str, experiment_name: str, reduced: bool = True):
    paths = load_project_paths()
    run_path = paths.run_dir(run_name)
    experiment_cfg, config_path = load_experiment_config(run_name, experiment_name)
    experiment_dir = paths.experiment_dir(run_name, experiment_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_memmap_dataset(run_name, reduced=reduced, soundscape=False)
    filenames = dataset.filenames
    feature_names = dataset.feature_names
    X = dataset.X
    y = dataset.y
    label_encoder_class_name = joblib.load(paths.label_encoders_dir / "class_name.joblib")
    label_encoder_primary_label = joblib.load(paths.label_encoders_dir / "primary_label.joblib")
    label_to_scope_mapping = np.load(paths.primary_to_class / "primary_to_class.npy")

    X_train = X
    y_train = y
    val_idx = None
    if experiment_cfg.training.train_val_split:
        train_idx, val_idx = get_idx_from_filenames(filenames)
        print(f"Size of train : {len(train_idx)}, size of val : {len(val_idx)}")
        X_train = X[train_idx]
        y_train = y[train_idx]

    estimator = build_estimator(experiment_cfg)
    x_val = None
    y_val = None
    if val_idx is not None:
        x_val = X[val_idx]
        y_val = y[val_idx]

    artifacts = train_dual_one_vs_rest_models(
        X_train,
        y_class_name=y_train[:, 0],
        y_primary_label=y_train[:, 1],
        label_encoder_class_name=label_encoder_class_name,
        label_encoder_primary_label=label_encoder_primary_label,
        class_name_estimator=estimator,
        primary_label_estimator=estimator,
        label_to_scope_mapping=label_to_scope_mapping,
        feature_names=feature_names,
        mil_mode=False,
        mil_config=None,
        batch_size=experiment_cfg.training.batch_size,
        epochs=experiment_cfg.training.epochs,
        early_stopping=experiment_cfg.training.early_stopping,
        n_iter_no_change=experiment_cfg.training.n_iter_no_change,
        tol=experiment_cfg.training.tol,
        x_val=x_val,
        y_class_name_val=None if y_val is None else y_val[:, 0],
        y_primary_label_val=None if y_val is None else y_val[:, 1],
        scope=experiment_cfg.training.scope,
    )

    stem = artifact_stem(experiment_cfg)
    if val_idx is not None:
        np.save(experiment_dir / "val_indices.npy", np.asarray(val_idx))

    joblib.dump(artifacts, experiment_dir / f"{stem}_ovr.joblib")
    with open(experiment_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment_cfg.to_dict(), handle, sort_keys=False)
    with open(experiment_dir / "config_source.txt", "w", encoding="utf-8") as handle:
        handle.write(str(config_path.resolve()))
    with open(experiment_dir / "run_source.txt", "w", encoding="utf-8") as handle:
        handle.write(str(run_path.resolve()))


def calibrate_and_save_ovr_models_chunks(
    run_name: str,
    experiment_name: str,
    *,
    calibration_name: str,
):
    paths = load_project_paths()
    experiment_cfg, _ = load_experiment_config(run_name, experiment_name)
    calibration_cfg, calibration_config_path = load_calibration_config(run_name, calibration_name)
    calibration_params = build_calibration_params(calibration_cfg)
    experiment_dir = paths.experiment_dir(run_name, experiment_name)
    stem = artifact_stem(experiment_cfg)

    val_idx_path = experiment_dir / "val_indices.npy"
    model_path = experiment_dir / f"{stem}_ovr.joblib"
    if not val_idx_path.exists():
        raise FileNotFoundError(
            f"Calibration requires saved validation indices: {val_idx_path}. "
            "Run training with training.train_val_split=true first."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained OVR artifacts not found: {model_path}")

    dataset = load_memmap_dataset(run_name, reduced=True, soundscape=False)
    val_idx = np.load(val_idx_path)
    x_val = dataset.X[val_idx]
    y_val = dataset.y[val_idx]
    artifacts = joblib.load(model_path)
    label_to_scope_mapping = np.load(paths.primary_to_class / "primary_to_class.npy")

    calibrated_artifacts = calibrate_dual_one_vs_rest_artifacts(
        artifacts,
        x_cal=x_val,
        y_class_name=y_val[:, 0],
        y_primary_label=y_val[:, 1],
        label_to_scope_mapping=label_to_scope_mapping,
        **calibration_params,
    )

    calibrated_probas = predict_proba_dual_one_vs_rest(calibrated_artifacts, x_val)
    calibrated_preds = predict_dual_one_vs_rest(calibrated_artifacts, x_val)
    calibration_stem = calibration_name.replace("/", "_")

    joblib.dump(calibrated_artifacts, experiment_dir / f"{stem}_ovr_calibrated_{calibration_stem}.joblib")
    np.save(experiment_dir / f"{stem}_val_probas_calibrated_{calibration_stem}.npy", calibrated_probas)
    np.save(experiment_dir / f"{stem}_val_preds_calibrated_{calibration_stem}.npy", calibrated_preds)
    with open(experiment_dir / f"calibration_config_source_{calibration_stem}.txt", "w", encoding="utf-8") as handle:
        handle.write(str(calibration_config_path.resolve()))


# def tune_and_save_ovr_thresholds(
#     run_name: str,
#     experiment_name: str,
#     *,
#     score_name: str = "macro_f1",
#     model_filename: str | None = None,
#     target_name: str = "class_name",
#     max_rounds: int = 2,
# ):
#     paths = load_project_paths()
#     experiment_cfg, _ = load_experiment_config(run_name, experiment_name)
#     experiment_dir = paths.experiment_dir(run_name, experiment_name)
#     stem = artifact_stem(experiment_cfg)

#     val_idx_path = experiment_dir / "val_indices.npy"
#     if not val_idx_path.exists():
#         raise FileNotFoundError(
#             f"Threshold tuning requires saved validation indices: {val_idx_path}. "
#             "Run training with training.train_val_split=true first."
#         )

#     artifact_filename = model_filename or f"{stem}_ovr.joblib"
#     model_path = experiment_dir / artifact_filename
#     if not model_path.exists():
#         raise FileNotFoundError(f"Saved OVR artifacts not found: {model_path}")

#     dataset = load_memmap_dataset(run_name)
#     val_idx = np.load(val_idx_path)
#     x_val = dataset.X[val_idx]
#     y_val = dataset.y[val_idx]
#     artifacts = joblib.load(model_path)

#     if isinstance(artifacts, DualOneVsRestArtifacts):
#         tuned_artifacts = tune_dual_one_vs_rest_thresholds(
#             artifacts,
#             x_val,
#             y_class_name=y_val[:, 0],
#             y_primary_label=y_val[:, 1],
#             score=score_name,
#             max_rounds=max_rounds,
#         )
#         tuned_preds = {
#             "class_name": predict_threshold_tuned_one_vs_rest(tuned_artifacts.class_name, x_val),
#             "primary_label": predict_threshold_tuned_one_vs_rest(tuned_artifacts.primary_label, x_val),
#         }
#         tuned_probas = predict_proba_dual_one_vs_rest(artifacts, x_val)
#         summary = {
#             "class_name": {
#                 "score_name": tuned_artifacts.class_name.score_name,
#                 "best_score": tuned_artifacts.class_name.best_score,
#                 "thresholds": tuned_artifacts.class_name.thresholds,
#             },
#             "primary_label": {
#                 "score_name": tuned_artifacts.primary_label.score_name,
#                 "best_score": tuned_artifacts.primary_label.best_score,
#                 "thresholds": tuned_artifacts.primary_label.thresholds,
#             },
#         }
#     elif isinstance(artifacts, OneVsRestArtifacts):
#         target_idx_mapping = {"class_name": 0, "primary_label": 1}
#         if target_name not in target_idx_mapping:
#             supported = ", ".join(sorted(target_idx_mapping))
#             raise ValueError(f"Unsupported target_name={target_name}. Supported: {supported}")
#         tuned_artifacts = tune_one_vs_rest_thresholds(
#             artifacts,
#             x_val,
#             y_true=y_val[:, target_idx_mapping[target_name]],
#             score=score_name,
#             max_rounds=max_rounds,
#         )
#         tuned_preds = predict_threshold_tuned_one_vs_rest(tuned_artifacts, x_val)
#         tuned_probas = predict_proba_one_vs_rest(artifacts, x_val)
#         summary = {
#             "score_name": tuned_artifacts.score_name,
#             "best_score": tuned_artifacts.best_score,
#             "thresholds": tuned_artifacts.thresholds,
#         }
#     else:
#         raise TypeError(
#             "Threshold tuning supports only OneVsRestArtifacts or DualOneVsRestArtifacts"
#         )

#     tuned_stem = Path(artifact_filename).stem
#     output_stem = f"{tuned_stem}_threshold_tuned_{score_name}"
#     joblib.dump(tuned_artifacts, experiment_dir / f"{output_stem}.joblib")
#     np.save(experiment_dir / f"{output_stem}_val_preds.npy", tuned_preds)
#     np.save(experiment_dir / f"{output_stem}_val_probas.npy", tuned_probas)
#     np.save(experiment_dir / f"{output_stem}_thresholds.npy", summary, allow_pickle=True)
