import os
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from time import time
import gc


from birdclef_2026_ml.feature_engineering.configs import ChunkConfig, MILConfig, FeatureConfig, PoolingConfig
from birdclef_2026_ml.models.one_vs_rest_models import train_dual_one_vs_rest_models
from birdclef_2026_ml.processing.data_split import split_audio_train_val
from birdclef_2026_ml.feature_engineering import build_feature_matrix_and_labels_from_df
from birdclef_2026_ml.models.mil_learning import build_mil_bags_from_df
from birdclef_2026_ml.paths import PATHS


def _save_config(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj.__dict__, f, indent=2)


def save_train_val_mil_bags_streaming(
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
    _save_config(chunk_cfg, config_dir / "chunk_config.json")
    _save_config(mil_cfg, config_dir / "mil_config.json")
    _save_config(feature_cfg, config_dir / "feature_config.json")
    _save_config(pooling_cfg, config_dir / "pooling_config.json")

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


def train_and_save_dual_one_vs_rest(
        x: np.ndarray,
        y_class_name: np.ndarray,
        y_primary_label: np.ndarray,
        class_name_estimator,
        primary_label_estimator,
        feature_names: list[str],
        run_dir: str,
        chunk_config: ChunkConfig = ChunkConfig(),
        mil_config: MILConfig = MILConfig(),
        feature_config: FeatureConfig = FeatureConfig(),
        pooling_config: PoolingConfig = PoolingConfig(),
        **kwargs
):
    """
    Train hierarchical dual one-vs-rest models and save all artifacts and configs.
    """

    os.makedirs(run_dir, exist_ok=True)

    # Train models
    dual_artifacts = train_dual_one_vs_rest_models(
        x=x,
        y_class_name=y_class_name,
        y_primary_label=y_primary_label,
        class_name_estimator=class_name_estimator,
        primary_label_estimator=primary_label_estimator,
        feature_names=feature_names,
        mil_mode=kwargs.get("mil_mode", False),
        mil_config=mil_config,
        **{k: v for k, v in kwargs.items() if k not in ["mil_mode", "mil_config"]}
    )

    # Save models
    joblib.dump(dual_artifacts.class_name, os.path.join(run_dir, "class_name_model.joblib"))
    joblib.dump(dual_artifacts.primary_label, os.path.join(run_dir, "primary_label_model.joblib"))

    # Save configs as JSON
    def save_config(obj, fname):
        with open(os.path.join(run_dir, fname), "w") as f:
            json.dump(obj.__dict__, f, indent=2)

    save_config(chunk_config, "chunk_config.json")
    save_config(mil_config, "mil_config.json")
    save_config(feature_config, "feature_config.json")
    save_config(pooling_config, "pooling_config.json")

    print(f"Models and configs saved to {run_dir}")
    return run_dir
