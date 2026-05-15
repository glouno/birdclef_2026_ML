
# Second stage :
# Features : P(c) | P(l) | P(c) * P(l) | SiteID | Time
import copy
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

from birdclef_2026_ml.feature_engineering.soundscapes import build_soundscape_context_features
from birdclef_2026_ml.processing.data_split import iter_soundscapes_oof_splits
from birdclef_2026_ml.paths import load_project_paths
from birdclef_2026_ml.models.artifacts import (
    DualOneVsRestArtifacts,
    MILSecondStageCleanAudioArtifacts,
)
from birdclef_2026_ml.models.mil import (
    assign_bag_ids,
    collapse_bag_labels,
    collapse_bag_multilabel_targets,
    bag_order,
)
from birdclef_2026_ml.configs import (
    load_experiment_config,
    artifact_stem,
)
from birdclef_2026_ml.processing.memmap_dataset import (
    load_memmap_dataset,
    load_ovr_inference_memmap,
)


def fit_second_stage_soundscapes(
    soundscapes: pd.DataFrame,
    run_name: str,
    experiment_name: str,
    *,
    reduced: bool = True,
    model_filename: str | None = None,
    batch_size: int | None = 65536,
    n_splits: int = 5,
):
    paths = load_project_paths()
    experiment_cfg, config_path = load_experiment_config(run_name, experiment_name)
    experiment_dir = paths.experiment_dir(run_name, experiment_name)
    soundscape_dir = experiment_dir / "soundscapes"
    soundscape_dir.mkdir(parents=True, exist_ok=True)

    stem = artifact_stem(experiment_cfg)
    dataset = load_memmap_dataset(run_name, reduced=reduced, soundscape=True)

    ss_context_features, _ = build_soundscape_context_features(
        soundscapes, filenames=dataset.filenames
    )

    output_stem = Path(model_filename or f"{stem}_ovr.joblib").stem
    output_stem = f"{output_stem}_soundscape"
    inference_dir = experiment_dir / "inference"
    inference_probas = load_ovr_inference_memmap(
        inference_dir,
        output_stem,
        dual=True,
    )
    class_proba = inference_probas.class_name
    primary_proba = inference_probas.primary_label
    if class_proba is None or primary_proba is None:
        raise FileNotFoundError("Missing dual-target inference memmaps")

    primary_to_class = np.asarray(
        np.load(paths.primary_to_class / "primary_to_class.npy"),
        dtype=int,
    )
    n_primary = int(primary_proba.shape[1])
    if primary_to_class.shape[0] < n_primary:
        raise ValueError("Primary-to-class mapping is shorter than primary labels")
    primary_to_class = primary_to_class[:n_primary]

    parent_proba = np.asarray(class_proba[:, primary_to_class], dtype=float)
    primary_proba = np.asarray(primary_proba[:, :n_primary], dtype=float)
    combined_proba = primary_proba * parent_proba

    if ss_context_features.shape[0] != primary_proba.shape[0]:
        raise ValueError("Context features do not align with soundscape rows")

    if dataset.y.ndim == 3:
        y_primary = np.asarray(dataset.y[:, 1, :], dtype=int)
    elif dataset.y.ndim == 2:
        y_primary = np.asarray(dataset.y, dtype=int)
    else:
        raise ValueError("Soundscape labels must be 2D or 3D")
    y_primary = y_primary[:, :n_primary]

    splits = iter_soundscapes_oof_splits(
        y=y_primary,
        filenames=np.asarray(dataset.filenames),
        n_splits=n_splits,
        shuffle=True,
    )

    oof_proba = primary_proba.copy()
    for fold_id, (train_idx, val_idx) in enumerate(splits):
        print(f"[SS second-stage] fold {fold_id + 1}/{n_splits}")
        if val_idx.size == 0:
            continue

        for class_id in range(n_primary):
            y_train = y_primary[train_idx, class_id]
            if np.unique(y_train).size < 2:
                oof_proba[val_idx, class_id] = primary_proba[val_idx, class_id]
                continue

            x_train = np.column_stack([
                primary_proba[train_idx, class_id],
                parent_proba[train_idx, class_id],
                combined_proba[train_idx, class_id],
                ss_context_features[train_idx],
            ])
            x_val = np.column_stack([
                primary_proba[val_idx, class_id],
                parent_proba[val_idx, class_id],
                combined_proba[val_idx, class_id],
                ss_context_features[val_idx],
            ])

            # pipe = Pipeline([
            #     ("scaler", StandardScaler()),
            #     ("model", LogisticRegression(
            #         max_iter=1000,
            #         solver="liblinear",
            #         class_weight="balanced",
            #     ))
            # ])
            sample_weight = compute_sample_weight("balanced", y_train)
            pipe = XGBClassifier(
                n_estimators=300,
                max_depth=3
            )
            pipe_fit = pipe.fit(x_train, y_train, sample_weight=sample_weight)
            pred_proba = pipe_fit.predict_proba(x_val)
            positive_col = int(
                # np.flatnonzero(pipe_fit.named_steps["model"].classes_ == 1)[0]
                np.flatnonzero(pipe_fit.classes_ == 1)[0]
            )
            oof_proba[val_idx, class_id] = pred_proba[:, positive_col]

    np.save(soundscape_dir / "oof_proba.npy", oof_proba)
    np.save(soundscape_dir / "combined_proba.npy", combined_proba)

    return soundscape_dir


def fit_second_stage_soundscapes_with_mil_proba(
    soundscapes: pd.DataFrame,
    run_name: str,
    experiment_name: str,
    *,
    reduced: bool = True,
    model_filename: str | None = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> Path:
    paths = load_project_paths()
    experiment_cfg, _ = load_experiment_config(run_name, experiment_name)
    experiment_dir = paths.experiment_dir(run_name, experiment_name)
    soundscape_dir = experiment_dir / "soundscapes"
    soundscape_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_memmap_dataset(run_name, reduced=reduced, soundscape=True)
    if dataset.bags_meta is None:
        raise ValueError("Soundscape MIL bag metadata is required")
    bag_meta_arr = np.asarray(dataset.bags_meta)
    if bag_meta_arr.ndim >= 2 and bag_meta_arr.shape[1] >= 1:
        bag_ids = np.asarray(bag_meta_arr[:, 0], dtype=int)
    elif bag_meta_arr.ndim == 1:
        bag_ids = np.asarray(bag_meta_arr, dtype=int)
    else:
        raise ValueError("Soundscape bag metadata must be 1D or 2D with bag ids in column 0")

    ordered_bag_ids, _ = bag_order(bag_ids)
    bag_filename_map: dict[int, str] = {}
    for bag_id, filename in zip(bag_ids, np.asarray(dataset.filenames)):
        bag_id_int = int(bag_id)
        if bag_id_int not in bag_filename_map:
            bag_filename_map[bag_id_int] = str(filename)
    bag_filenames = np.asarray(
        [bag_filename_map[int(bag_id)] for bag_id in ordered_bag_ids],
        dtype=object,
    )
    ss_context_features, _ = build_soundscape_context_features(
        soundscapes, bag_filenames
    )

    stem = artifact_stem(experiment_cfg)
    output_stem = Path(model_filename or f"{stem}_ovr.joblib").stem
    clean_dir = experiment_dir / "clean_audio"

    mil_proba_path = clean_dir / f"{output_stem}_mil_second_stage_primary_soundscape_proba.npy"
    if not mil_proba_path.exists():
        raise FileNotFoundError(f"Soundscape MIL proba not found: {mil_proba_path}")
    mil_proba = np.asarray(np.load(mil_proba_path), dtype=float)
    bag_features_path = clean_dir / "train_bag_features_primary_soundscape.npy"
    if not bag_features_path.exists():
        raise FileNotFoundError(f"Soundscape bag features not found: {bag_features_path}")
    bag_features = np.load(bag_features_path, mmap_mode="r")

    n_classes = int(mil_proba.shape[1])
    if bag_features.shape[0] != n_classes:
        raise ValueError("Soundscape bag features do not align with MIL probas")
    if mil_proba.shape[0] != ordered_bag_ids.shape[0]:
        raise ValueError("MIL probas do not align with soundscape bag ids")
    if bag_features.shape[1] != ordered_bag_ids.shape[0]:
        raise ValueError("Soundscape bag features do not align with bag ids")
    if ss_context_features.shape[0] != ordered_bag_ids.shape[0]:
        raise ValueError("Context features do not align with bag ids")

    if dataset.y.ndim == 3:
        y_primary = np.asarray(dataset.y[:, 1, :], dtype=int)
    elif dataset.y.ndim == 2:
        y_primary = np.asarray(dataset.y, dtype=int)
    else:
        raise ValueError("Soundscape labels must be 2D or 3D")
    y_primary = y_primary[:, :n_classes]
    bag_targets = collapse_bag_multilabel_targets(bag_ids, y_primary)

    splits = iter_soundscapes_oof_splits(
        y=bag_targets,
        filenames=bag_filenames,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    base_features = np.concatenate([mil_proba, ss_context_features], axis=1)
    oof_proba = mil_proba.copy()
    for fold_id, (train_idx, val_idx) in enumerate(splits):
        print(f"[MIL+context] fold {fold_id + 1}/{len(splits)}")
        if val_idx.size == 0:
            continue

        for class_id in range(n_classes):
            y_train = bag_targets[train_idx, class_id]
            if np.unique(y_train).size < 2:
                oof_proba[val_idx, class_id] = mil_proba[val_idx, class_id]
                continue

            x_train = np.concatenate(
                [base_features[train_idx], bag_features[class_id, train_idx]],
                axis=1,
            )
            x_val = np.concatenate(
                [base_features[val_idx], bag_features[class_id, val_idx]],
                axis=1,
            )

            pipe = XGBClassifier(
                n_estimators=300,
                max_depth=3
            )
            sample_weight = compute_sample_weight("balanced", y_train)
            pipe_fit = pipe.fit(x_train, y_train, sample_weight=sample_weight)
            pred_proba = pipe_fit.predict_proba(x_val)
            positive_col = int(
                # .named_steps["model"]
                np.flatnonzero(pipe_fit.classes_ == 1)[0]
            )
            oof_proba[val_idx, class_id] = pred_proba[:, positive_col]

    output_path = soundscape_dir / f"{output_stem}_mil_second_stage_primary_soundscape_context_oof.npy"
    np.save(output_path, oof_proba)
    return output_path


def build_mil_second_stage_clean_audio_data(
    run_name: str,
    experiment_name: str,
    *,
    soundscape: bool = False,
    reduced: bool = True,
    model_filename: str | None = None,
    batch_size: int | None = 65536,
    bag_batch_size: int | None = 512,
):
    paths = load_project_paths()
    experiment_cfg, _ = load_experiment_config(run_name, experiment_name)
    experiment_dir = paths.experiment_dir(run_name, experiment_name)
    clean_dir = experiment_dir / "clean_audio"
    clean_dir.mkdir(parents=True, exist_ok=True)

    stem = artifact_stem(experiment_cfg)
    model_path = experiment_dir / (model_filename or f"{stem}_ovr.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Saved OVR artifacts not found: {model_path}")

    dataset = load_memmap_dataset(run_name, reduced=reduced, soundscape=soundscape)

    artifacts = joblib.load(model_path)
    if not isinstance(artifacts, DualOneVsRestArtifacts):
        raise TypeError("Clean-audio second stage requires DualOneVsRestArtifacts")

    if dataset.bags_meta is None:
        raise ValueError("Clean-audio second stage requires MIL bag metadata")
    bag_meta_arr = np.asarray(dataset.bags_meta)
    if soundscape:
        if bag_meta_arr.ndim >= 2 and bag_meta_arr.shape[1] >= 1:
            bag_ids = np.asarray(bag_meta_arr[:, 0], dtype=int)
        elif bag_meta_arr.ndim == 1:
            bag_ids = np.asarray(bag_meta_arr, dtype=int)
        else:
            raise ValueError(
                "Soundscape bag metadata must be 1D or 2D with bag ids in column 0"
            )
    else:
        if bag_meta_arr.ndim != 1:
            raise ValueError("Clean-audio bag metadata must be 1D window ids")
        bag_ids = assign_bag_ids(bag_meta_arr, run_name)

    all_idx = np.arange(dataset.X.shape[0], dtype=int)
    if soundscape:
        train_idx = all_idx
        val_idx = np.empty(0, dtype=int)
    else:
        val_idx_path = experiment_dir / "val_indices.npy"
        if not val_idx_path.exists():
            raise FileNotFoundError(f"Validation indices not found: {val_idx_path}")
        val_idx = np.asarray(np.load(val_idx_path), dtype=int)
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)

    suffix = "_soundscape" if soundscape else ""
    ordered_bag_ids_all, _ = bag_order(bag_ids)
    print(
        "[MIL clean] windows={}, bags={}, train_windows={}, val_windows={}, bag_batch_size={}".format(
            bag_ids.shape[0],
            ordered_bag_ids_all.shape[0],
            train_idx.shape[0],
            val_idx.shape[0],
            bag_batch_size,
        )
    )

    primary_to_class = np.asarray(np.load(paths.primary_to_class / "primary_to_class.npy"), dtype=int)

    inference_dir = experiment_dir / "inference"
    output_stem = Path(model_filename or f"{stem}_ovr.joblib").stem
    if soundscape:
        output_stem = f"{output_stem}_soundscape"
    inference_probas = load_ovr_inference_memmap(
        inference_dir,
        output_stem,
        dual=True,
    )
    class_proba = inference_probas.class_name
    primary_proba = inference_probas.primary_label
    if class_proba is None or primary_proba is None:
        raise FileNotFoundError("Missing dual-target inference memmaps")
    n_primary = int(primary_proba.shape[1])
    n_features = 19

    def _series_features(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros(8, dtype=float)
        mean_v = float(np.mean(values))
        max_v = float(np.max(values))
        std_v = float(np.std(values))
        min_v = float(np.min(values))
        argmax_v = float(np.argmax(values))
        argmin_v = float(np.argmin(values))
        if values.size < 2:
            mean_abs_diff = 0.0
        else:
            mean_abs_diff = float(np.mean(np.abs(np.diff(values))))
        p80 = float(np.percentile(values, 80))
        pct_above_p80 = float(np.mean(values > p80))
        return np.array(
            [
                mean_v,
                max_v,
                std_v,
                min_v,
                argmax_v,
                argmin_v,
                mean_abs_diff,
                pct_above_p80,
            ],
            dtype=float,
        )

    def _pair_features(
        primary_series: np.ndarray,
        parent_series: np.ndarray,
        mean_primary: float,
        mean_parent: float,
    ) -> np.ndarray:
        mean_diff = mean_primary - mean_parent
        if primary_series.size < 2:
            corr = 0.0
        else:
            std_primary = float(np.std(primary_series))
            std_parent = float(np.std(parent_series))
            if std_primary == 0.0 or std_parent == 0.0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(primary_series, parent_series)[0, 1])
        ratio = np.divide(
            primary_series,
            parent_series,
            out=np.zeros_like(primary_series),
            where=parent_series > 0,
        )
        mean_ratio = float(np.mean(ratio)) if ratio.size > 0 else 0.0
        return np.array([mean_diff, corr, mean_ratio], dtype=float)

    def _build_bag_features(
        subset_idx: np.ndarray,
        subset_bag_ids: np.ndarray,
        subset_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        ordered_bag_ids, first_idx = bag_order(subset_bag_ids)
        if ordered_bag_ids.size == 0:
            return np.empty((0, 0, 0), dtype=float), ordered_bag_ids

        feature_memmap = np.lib.format.open_memmap(
            clean_dir / f"{subset_name}_bag_features_primary{suffix}.npy",
            mode="w+",
            dtype=float,
            shape=(n_primary, ordered_bag_ids.shape[0], n_features),
        )
        bag_idx_by_id = {int(bid): idx for idx, bid in enumerate(ordered_bag_ids)}
        end_idx = np.concatenate([first_idx[1:], [subset_bag_ids.shape[0]]])

        batch_size_bags = bag_batch_size or ordered_bag_ids.shape[0]
        n_batches = (ordered_bag_ids.shape[0] + batch_size_bags - 1) // batch_size_bags
        print("[MIL clean] {} bags={}, windows={}, batches={}".format(
            subset_name,
            ordered_bag_ids.shape[0],
            subset_bag_ids.shape[0],
            n_batches,
        )
        )
        for start in range(0, ordered_bag_ids.shape[0], batch_size_bags):
            stop = min(start + batch_size_bags, ordered_bag_ids.shape[0])
            batch_bag_ids = ordered_bag_ids[start:stop]
            batch_id = start // batch_size_bags + 1
            if batch_id == 1 or batch_id == n_batches or batch_id % 25 == 0:
                print(f"[MIL clean] {subset_name} batch {batch_id}/{n_batches}")
            idx_start = int(first_idx[start])
            idx_stop = int(end_idx[stop - 1])
            bag_ids_batch = subset_bag_ids[idx_start:idx_stop]

            if np.all(np.isin(bag_ids_batch, batch_bag_ids)):
                window_idx = subset_idx[idx_start:idx_stop]
            else:
                mask = np.isin(subset_bag_ids, batch_bag_ids)
                window_idx = subset_idx[mask]
                bag_ids_batch = subset_bag_ids[mask]

            class_batch = np.asarray(class_proba[window_idx], dtype=float)
            primary_batch = np.asarray(primary_proba[window_idx], dtype=float)

            batch_ordered_bag_ids, _ = bag_order(bag_ids_batch)
            print(len(batch_ordered_bag_ids))
            for bag_id in batch_ordered_bag_ids:
                bag_mask = bag_ids_batch == bag_id
                bag_window_idx = window_idx[bag_mask]
                bag_primary = primary_batch[bag_mask]
                bag_class = class_batch[bag_mask]
                bag_pos = bag_idx_by_id[int(bag_id)]

                for primary_id in range(n_primary):
                    parent_id = int(primary_to_class[primary_id])
                    primary_series = bag_primary[:, primary_id]
                    parent_series = bag_class[:, parent_id]

                    primary_stats = _series_features(primary_series)
                    parent_stats = _series_features(parent_series)
                    pair_stats = _pair_features(
                        primary_series,
                        parent_series,
                        mean_primary=float(primary_stats[0]),
                        mean_parent=float(parent_stats[0]),
                    )
                    feature_memmap[primary_id, bag_pos] = np.concatenate(
                        [primary_stats, parent_stats, pair_stats],
                        axis=0,
                    )

        feature_memmap.flush()
        return feature_memmap, ordered_bag_ids

    bag_features_train, ordered_bag_ids_train = _build_bag_features(
        train_idx,
        bag_ids[train_idx],
        "train",
    )
    bag_features_val, ordered_bag_ids_val = _build_bag_features(
        val_idx,
        bag_ids[val_idx],
        "val",
    )
    print("Number of features", bag_features_train.shape)

    bag_labels_train = collapse_bag_labels(bag_ids[train_idx], dataset.y[train_idx, 1])
    bag_labels_val = collapse_bag_labels(bag_ids[val_idx], dataset.y[val_idx, 1])
    np.save(clean_dir / f"train_bag_labels{suffix}.npy", bag_labels_train)
    np.save(clean_dir / f"val_bag_labels{suffix}.npy", bag_labels_val)
    np.save(clean_dir / f"train_bag_ids{suffix}.npy", ordered_bag_ids_train)
    np.save(clean_dir / f"val_bag_ids{suffix}.npy", ordered_bag_ids_val)

    return clean_dir


def train_mil_second_stage_clean_audio(
    run_name: str,
    experiment_name: str,
    *,
    model_filename: str | None = None,
):
    paths = load_project_paths()
    experiment_cfg, _ = load_experiment_config(run_name, experiment_name)
    experiment_dir = paths.experiment_dir(run_name, experiment_name)
    clean_dir = experiment_dir / "clean_audio"

    stem = artifact_stem(experiment_cfg)
    model_path = experiment_dir / (model_filename or f"{stem}_ovr.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Saved OVR artifacts not found: {model_path}")

    base_artifacts = joblib.load(model_path)
    if not isinstance(base_artifacts, DualOneVsRestArtifacts):
        raise TypeError("Clean-audio second stage requires DualOneVsRestArtifacts")

    bag_features_train = np.load(
        clean_dir / "train_bag_features_primary.npy", mmap_mode="r"
    )
    bag_features_val = np.load(
        clean_dir / "val_bag_features_primary.npy", mmap_mode="r"
    )
    bag_labels_train = np.load(clean_dir / "train_bag_labels.npy")

    n_classes = len(base_artifacts.primary_label.label_encoder.classes_)
    if bag_features_train.shape[0] != n_classes:
        raise ValueError(
            "Primary-label feature count does not match label encoder classes"
        )
    val_proba = np.zeros((bag_features_val.shape[1], n_classes), dtype=float)

    estimators: list[Pipeline | None] = [None] * n_classes
    fallback_positive_probs: list[float | None] = [None] * n_classes
    skipped_class_ids: list[int] = []

    for class_id in range(n_classes):
        print(class_id)
        x_train = np.asarray(bag_features_train[class_id], dtype=float)
        x_val = np.asarray(bag_features_val[class_id], dtype=float)
        y_binary = (bag_labels_train == class_id).astype(int)
        if np.unique(y_binary).size < 2:
            print(f"Skipping class {class_id}")
            fallback_prob = float(np.mean(y_binary))
            fallback_positive_probs[class_id] = fallback_prob
            skipped_class_ids.append(class_id)
            val_proba[:, class_id] = fallback_prob
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced"
            ))
        ])
        pipe_fit = pipe.fit(x_train, y_binary)
        estimators[class_id] = pipe_fit
        print("Fitted classifier")
        pred_proba = pipe_fit.predict_proba(x_val)
        positive_col = int(np.flatnonzero(pipe_fit.named_steps["model"].classes_ == 1)[0])
        val_proba[:, class_id] = pred_proba[:, positive_col]

    np.save(clean_dir / "val_proba.npy", val_proba)
    output_stem = Path(model_filename or f"{stem}_ovr.joblib").stem
    artifact_path = clean_dir / f"{output_stem}_mil_second_stage_primary.joblib"
    training_summary = {
        "trained_classes": int(n_classes - len(skipped_class_ids)),
        "skipped_classes": int(len(skipped_class_ids)),
        "skipped_class_ids": skipped_class_ids,
    }
    second_stage_artifacts = MILSecondStageCleanAudioArtifacts(
        estimators=estimators,
        label_encoder=base_artifacts.primary_label.label_encoder,
        fallback_positive_probs=fallback_positive_probs,
        training_summary=training_summary,
    )
    joblib.dump(second_stage_artifacts, artifact_path)
    print(f"Saved clean-audio second-stage artifacts to {artifact_path}")
    return clean_dir


def run_mil_second_stage_clean_audio_soundscape_inference(
    run_name: str,
    experiment_name: str,
    *,
    model_filename: str | None = None,
) -> Path:
    paths = load_project_paths()
    experiment_cfg, _ = load_experiment_config(run_name, experiment_name)
    experiment_dir = paths.experiment_dir(run_name, experiment_name)
    clean_dir = experiment_dir / "clean_audio"

    stem = artifact_stem(experiment_cfg)
    output_stem = Path(model_filename or f"{stem}_ovr.joblib").stem
    artifact_path = clean_dir / f"{output_stem}_mil_second_stage_primary.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Second-stage artifacts not found: {artifact_path}")

    artifacts = joblib.load(artifact_path)
    if not isinstance(artifacts, MILSecondStageCleanAudioArtifacts):
        raise TypeError("Expected MILSecondStageCleanAudioArtifacts")

    features_path = clean_dir / "train_bag_features_primary_soundscape.npy"
    if not features_path.exists():
        raise FileNotFoundError(f"Soundscape bag features not found: {features_path}")
    bag_features = np.load(features_path, mmap_mode="r")

    n_classes = len(artifacts.label_encoder.classes_)
    if bag_features.shape[0] != n_classes:
        raise ValueError(
            "Soundscape bag feature count does not match label encoder classes"
        )

    proba = np.zeros((bag_features.shape[1], n_classes), dtype=float)
    for class_id in range(n_classes):
        estimator = artifacts.estimators[class_id]
        fallback_prob = artifacts.fallback_positive_probs[class_id]
        if estimator is None:
            if fallback_prob is None:
                raise ValueError("Missing fallback probability for untrained estimator")
            proba[:, class_id] = float(fallback_prob)
            continue

        x = np.asarray(bag_features[class_id], dtype=float)
        pred = estimator.predict_proba(x)
        if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
            classes = estimator.named_steps["model"].classes_
        else:
            classes = estimator.classes_
        positive_col = int(np.flatnonzero(np.asarray(classes) == 1)[0])
        proba[:, class_id] = pred[:, positive_col]

    output_path = clean_dir / f"{output_stem}_mil_second_stage_primary_soundscape_proba.npy"
    np.save(output_path, proba)
    return output_path


def fit_mil_second_stage_clean_audio(
    run_name: str,
    experiment_name: str,
    *,
    soundscape: bool = False,
    reduced: bool = True,
    model_filename: str | None = None,
    batch_size: int | None = 65536,
    bag_batch_size: int | None = 512,
):
    build_mil_second_stage_clean_audio_data(
        run_name,
        experiment_name,
        soundscape=soundscape,
        reduced=reduced,
        model_filename=model_filename,
        batch_size=batch_size,
        bag_batch_size=bag_batch_size,
    )
    return train_mil_second_stage_clean_audio(
        run_name,
        experiment_name,
        model_filename=model_filename,
    )
