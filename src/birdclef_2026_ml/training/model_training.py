
import numpy as np
import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight

from birdclef_2026_ml.processing.data_split import split_audio_train_val
from birdclef_2026_ml.models.one_vs_rest_models import (
    train_one_vs_rest_model,
    train_dual_one_vs_rest_models,
    predict_proba_one_vs_rest,
    predict_one_vs_rest
)
from birdclef_2026_ml.paths import PATHS


def train_and_save_ovr_models_chunks(run_name: str, train_val_split: bool):
    filenames = np.load(PATHS["models"] / run_name / "file_ids.npy", allow_pickle=True)
    feature_names = np.load(PATHS["models"] / run_name / "feature_names.npy", allow_pickle=True)
    y_shape = tuple(np.load(PATHS["models"] / run_name / "shape_y.npy"))
    X_shape = tuple(np.load(PATHS["models"] / run_name / "shape_X.npy"))
    y_dtype = np.load(PATHS["models"] / run_name / "dtype_y.npy").item()
    X_dtype = np.load(PATHS["models"] / run_name / "dtype_X.npy").item()
    X = np.memmap(PATHS["models"] / run_name / "X.dat",
                  dtype=X_dtype, shape=X_shape)
    y = np.memmap(PATHS["models"] / run_name / "y.dat",
                  dtype=y_dtype, shape=y_shape)

    label_encoder_class_name = joblib.load(PATHS["label_encoders"] / "class_name.joblib")
    label_encoder_primary_label = joblib.load(PATHS["label_encoders"] / "primary_label.joblib")

    X_train, y_train = X, y
    X_val, y_val = None, None
    if train_val_split:
        train_idx, val_idx = split_audio_train_val(X, y, filenames)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

    # classes = np.unique(y_train)
    # class_weights = compute_class_weight(
    #     class_weight="balanced",
    #     classes=classes,
    #     y=np.asarray(y_train).ravel()
    # )
    # class_weight_dict = dict(zip(classes, class_weights))
    print(y)
    estimator = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        max_iter=3000,
        tol=1e-4,
        early_stopping=False,
        learning_rate="optimal",
        average=True,
        random_state=42
    )

    artifacts = train_one_vs_rest_model(
        X_train,
        y_enc=y_train[:, 0],
        label_encoder=label_encoder_class_name,
        estimator=estimator,
        feature_names=feature_names,
        class_scope=None,
        feature_reweighting=False,
        profile_target="class_name",
        alpha=1,
        mil_mode=False,
        mil_config=None,
        batch_size=128,
        epochs=3
    )

    if train_val_split:
        probas = predict_proba_one_vs_rest(artifacts, X_val)
        preds = predict_one_vs_rest(artifacts, X_val)

        np.save(PATHS["models"] / run_name / "probas.npy", probas)
        np.save(PATHS["models"] / run_name / "preds.npy", preds)
        np.save(PATHS["models"] / run_name / "true.npy", np.asarray(y_val))

    # artifacts = train_dual_one_vs_rest_models(X_train,
    #                                           y_class_name=y_train[:, 0],
    #                                           y_primary_label=y_train[:, 1],
    #                                           label_encoder_class_name=label_encoder_class_name,
    #                                           label_encoder_primary_label=label_encoder_primary_label,
    #                                           class_name_estimator=estimator,
    #                                           primary_label_estimator=estimator,
    #                                           feature_names=feature_names,
    #                                           feature_reweighting=True,
    #                                           batch_size=16)

    joblib.dump(artifacts, PATHS["models"] / run_name / "sgdclassifier_logloss_class_name.joblib")
