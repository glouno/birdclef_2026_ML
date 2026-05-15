import joblib
import numpy as np
import yaml

from birdclef_2026_ml.configs import (
    load_experiment_config,
    artifact_stem,
    build_estimator
)
from birdclef_2026_ml.models.one_vs_rest import train_dual_one_vs_rest_models
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
