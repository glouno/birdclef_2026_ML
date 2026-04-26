import os
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from time import time
import gc
import shutil


from birdclef_2026_ml.configs import ChunkConfig, MILConfig, FeatureConfig, PoolingConfig
from birdclef_2026_ml.models.one_vs_rest_models import (
    DualOneVsRestArtifacts,
    train_one_vs_rest_model,
    train_dual_one_vs_rest_models
)
from birdclef_2026_ml.processing.data_split import split_audio_train_val
from birdclef_2026_ml.models.mil_learning import build_mil_bags_from_df
from birdclef_2026_ml.paths import PATHS
from birdclef_2026_ml.processing.audio_utils import load_config, save_config


def build_feature_matrices(
    df: pd.DataFrame,
    feature_cfg=FeatureConfig(),
    pooling_cfg=PoolingConfig(),
    chunk_cfg=ChunkConfig(),
    mil_cfg=MILConfig(),
    run_dir: Path = PATHS["models"] / "runs",
    n_splits: int = 50,
    pathroot: str = "train_audio_spectral_gating_dir"
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = run_dir / timestamp
    data_dir = base_dir / "data"
    config_dir = base_dir / "config"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Save configs
    save_config(chunk_cfg, config_dir / "mil_bags" / "chunk_config.json")
    save_config(mil_cfg, config_dir / "mil_bags" / "mil_config.json")
    save_config(feature_cfg, config_dir / "mil_bags" / "feature_config.json")
    save_config(pooling_cfg, config_dir / "mil_bags" / "pooling_config.json")

    # Split
    df_train, df_val = split_audio_train_val(df)

    def process_and_stream(df_part, out_dir, prefix):
        idx_map = []
        feature_names = None

        chunk_size = max(1, len(df_part) // n_splits)

        for i in range(0, len(df_part), chunk_size):
            chunk_df = df_part.iloc[i:i + chunk_size]

            print(f"[{prefix}] chunk {i // chunk_size + 1}")

            bags, idx_expanded, feat_names = build_mil_bags_from_df(
                df=chunk_df,
                feature_cfg=feature_cfg,
                pooling_cfg=pooling_cfg,
                chunk_cfg=chunk_cfg,
                mil_cfg=mil_cfg,
                pathroot=pathroot,
            )

            if feature_names is None:
                feature_names = feat_names

            joblib.dump(
                bags,
                out_dir / f"bags_{i // chunk_size:04d}.joblib",
                compress=3
            )

            idx_map.extend(idx_expanded)

            # Release memory immediately
            del bags
            del idx_expanded
            gc.collect()

        return idx_map, feature_names

    print("Processing train set...")

    start = time()
    idx_train, feature_names = process_and_stream(df_train, train_dir, "train")
    print(f"Train done in {time() - start:.1f}s")
    print("Processing val set...")
    idx_val, _ = process_and_stream(df_val, val_dir, "val")
    # Save label arrays separately (small)

    joblib.dump(
        df_train.loc[idx_train, ["class_name", "primary_label"]].values,
        data_dir / "train_labels.joblib",
        compress=3
    )

    joblib.dump(
        df_val.loc[idx_val, ["class_name", "primary_label"]].values,
        data_dir / "val_labels.joblib",
        compress=3
    )

    # Save index mapping
    joblib.dump(
        {"idx_train": idx_train, "idx_val": idx_val, "feature_names": feature_names},
        data_dir / "meta.joblib",
        compress=3

    )

    print(f"Saved streaming dataset to: {data_dir}")

    return data_dir


def _resolve_streaming_run_dirs(run_dir: str | Path) -> tuple[Path, Path, Path]:
    base_dir = Path(run_dir)
    if base_dir.name == "data" and (base_dir.parent / "config").exists():
        base_dir = base_dir.parent

    data_dir = base_dir / "data"
    config_dir = base_dir / "config"
    if not data_dir.exists():
        raise FileNotFoundError(f"Streaming data directory not found: {data_dir}")
    if not config_dir.exists():
        raise FileNotFoundError(f"Streaming config directory not found: {config_dir}")

    return base_dir, data_dir, config_dir


def _iter_streaming_bag_paths(split_dir: Path) -> list[Path]:
    bag_paths = sorted(split_dir.glob("bags_*.joblib"))
    if not bag_paths:
        raise FileNotFoundError(f"No streamed MIL bag shards found in {split_dir}")
    return bag_paths


def _memmap_string_dtype(values: np.ndarray) -> str:
    values_arr = np.asarray(values, dtype=str)
    max_len = max(1, max(len(v) for v in values_arr.ravel()))
    return f"<U{max_len}"


def _prepare_streaming_training_memmaps(
    train_dir: Path,
    train_labels_path: Path,
    temp_dir: Path,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    bag_paths = _iter_streaming_bag_paths(train_dir)
    train_labels = np.asarray(joblib.load(train_labels_path))
    if train_labels.ndim != 2 or train_labels.shape[1] != 2:
        raise ValueError(
            "train_labels.joblib must contain a 2D array with columns [class_name, primary_label]"
        )

    n_bags_seen = 0
    n_instances = 0
    n_features: int | None = None

    for bag_path in bag_paths:
        bags = joblib.load(bag_path)
        n_bags_seen += len(bags)
        for bag in bags:
            bag_arr = np.asarray(bag, dtype=float)
            if bag_arr.ndim == 1:
                bag_arr = bag_arr[np.newaxis, :]
            if bag_arr.ndim != 2:
                raise ValueError(f"Each MIL bag must be 2D, got shape={bag_arr.shape} in {bag_path}")
            if n_features is None:
                n_features = int(bag_arr.shape[1])
            elif bag_arr.shape[1] != n_features:
                raise ValueError(
                    f"Inconsistent feature count in {bag_path}: {bag_arr.shape[1]} != {n_features}"
                )
            n_instances += int(bag_arr.shape[0])
        del bags
        gc.collect()

    if n_bags_seen != len(train_labels):
        raise ValueError(
            "Mismatch between streamed bag count and saved train labels "
            f"({n_bags_seen} != {len(train_labels)})"
        )
    if n_features is None:
        raise ValueError("No features found in streamed MIL bags")

    x_memmap = np.memmap(
        temp_dir / "train_x.dat",
        dtype=np.float32,
        mode="w+",
        shape=(n_instances, n_features),
    )
    y_class_memmap = np.memmap(
        temp_dir / "train_y_class.dat",
        dtype=_memmap_string_dtype(train_labels[:, 0]),
        mode="w+",
        shape=(n_instances,),
    )
    y_primary_memmap = np.memmap(
        temp_dir / "train_y_primary.dat",
        dtype=_memmap_string_dtype(train_labels[:, 1]),
        mode="w+",
        shape=(n_instances,),
    )

    instance_offset = 0
    bag_offset = 0
    for bag_path in bag_paths:
        bags = joblib.load(bag_path)
        for bag in bags:
            bag_arr = np.asarray(bag, dtype=np.float32)
            if bag_arr.ndim == 1:
                bag_arr = bag_arr[np.newaxis, :]

            next_offset = instance_offset + bag_arr.shape[0]
            x_memmap[instance_offset:next_offset] = bag_arr
            y_class_memmap[instance_offset:next_offset] = str(train_labels[bag_offset, 0])
            y_primary_memmap[instance_offset:next_offset] = str(train_labels[bag_offset, 1])

            instance_offset = next_offset
            bag_offset += 1

        del bags
        gc.collect()

    x_memmap.flush()
    y_class_memmap.flush()
    y_primary_memmap.flush()
    return x_memmap, y_class_memmap, y_primary_memmap


def train_and_save_dual_one_vs_rest(
        run_dir: str | Path,
        class_name_estimator,
        primary_label_estimator,
):
    """
    Train hierarchical dual one-vs-rest models from a streamed MIL run directory.

    The directory must already contain the `data/` and `config/` artifacts produced
    by `save_train_val_mil_bags_streaming`. This function loads the saved config and
    streamed train bags, fits the models from disk-backed memmaps, and saves only the
    trained model artifacts with joblib.
    """
    base_dir, data_dir, config_dir = _resolve_streaming_run_dirs(run_dir)
    mil_config = MILConfig(**load_config(config_dir / "mil_config.json"))
    meta = joblib.load(data_dir / "meta.joblib")
    feature_names = list(meta["feature_names"])

    temp_dir = base_dir / "tmp_training_memmap"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        x_train, y_class_name, y_primary_label = _prepare_streaming_training_memmaps(
            train_dir=data_dir / "train",
            train_labels_path=data_dir / "train_labels.joblib",
            temp_dir=temp_dir,
        )

        # base_estimator = LogisticRegression(max_iter=1000)
        # shared_kwargs = {k: v for k, v in kwargs.items() if k not in {"mil_mode", "mil_config"}}
        dual_artifacts = train_dual_one_vs_rest_models(x_train,
                                                       y_class_name,
                                                       y_primary_label,
                                                       class_name_estimator,
                                                       primary_label_estimator,
                                                       feature_names,
                                                       True,
                                                       None,
                                                       True,
                                                       mil_config)

        # class_name_artifacts = train_one_vs_rest_model(
        #     x=x_train,
        #     y=y_class_name,
        #     estimator=class_name_estimator,
        #     feature_names=feature_names,
        #     profile_target="class_name",
        #     mil_mode=True,
        #     mil_config=mil_config,
        #     **shared_kwargs,
        # )
        # primary_label_artifacts = train_one_vs_rest_model(
        #     x=x_train,
        #     y=y_primary_label,
        #     estimator=primary_label_estimator,
        #     feature_names=feature_names,
        #     class_scope=y_class_name,
        #     profile_weights_by_label=profile_target="primary_label",
        #     mil_mode=FalTrse,
        #     mil_config=None,
        #     **shared_kwargs,
        # )

        # class_name_artifacts.mil_mode = True
        # class_name_artifacts.mil_config = mil_config
        # primary_label_artifacts.mil_mode = True
        # primary_label_artifacts.mil_config = mil_config

        # dual_artifacts = DualOneVsRestArtifacts(
        #     class_name=class_name_artifacts,
        #     primary_label=primary_label_artifacts,
        # )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Save models
    joblib.dump(dual_artifacts.class_name, base_dir / "class_name_model.joblib")
    joblib.dump(dual_artifacts.primary_label, base_dir / "primary_label_model.joblib")

    print(f"Models saved to {base_dir}")
    return str(base_dir)
