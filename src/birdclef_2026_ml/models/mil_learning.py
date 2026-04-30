
from typing import Any, Callable, Iterable
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

from birdclef_2026_ml.feature_engineering import build_mil_feature_matrix, split_audio_into_chunks
from birdclef_2026_ml.configs import PipelineConfig, MILConfig
from birdclef_2026_ml.processing.audio_utils import build_audio_path, get_path, load_audio
from birdclef_2026_ml.feature_engineering.chunking import get_sliding_window_intervals
from birdclef_2026_ml.feature_engineering.utils import joblib_to_frame_features
from birdclef_2026_ml.models.artifacts import MILFeatureBags, Array1D, Array2D


def _validate_mil_enabled(mil_mode: bool, mil_config: MILConfig | None) -> None:
    if mil_mode and mil_config is None:
        raise ValueError("mil_config is required when mil_mode=True")


def _coerce_bags(x: Any) -> tuple[list[Array2D], Array1D | None]:
    if isinstance(x, MILFeatureBags):
        return list(x.bags), None if x.bag_ids is None else np.asarray(x.bag_ids)

    bags = list(x)
    out: list[Array2D] = []
    for bag in bags:
        bag_arr = np.asarray(bag, dtype=float)
        if bag_arr.ndim == 1:
            bag_arr = bag_arr[np.newaxis, :]
        if bag_arr.ndim != 2:
            raise ValueError("Each MIL bag must be a 2D array-like object")
        out.append(bag_arr)
    return out, None


def flatten_mil_bags(
    x: Any,
    y: Any | None = None,
) -> tuple[Array2D, Array1D | None, Array1D]:
    """Flatten bags into one instance matrix and optionally repeat bag labels."""
    bags, _ = _coerce_bags(x)
    if not bags:
        x_empty = np.empty((0, 0), dtype=float)
        y_empty = None if y is None else np.array([], dtype=np.asarray(y).dtype)
        return x_empty, y_empty, np.array([], dtype=int)

    bag_sizes = np.asarray([bag.shape[0] for bag in bags], dtype=int)
    if np.any(bag_sizes <= 0):
        raise ValueError("Each MIL bag must contain at least one instance")

    x_flat = np.vstack(bags)
    y_flat = None
    if y is not None:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array-like input")
        if y_arr.shape[0] != len(bags):
            raise ValueError(
                "MIL labels must contain one label per bag "
                f"({y_arr.shape[0]} != {len(bags)})"
            )
        y_flat = np.repeat(y_arr, bag_sizes)

    return x_flat, y_flat, bag_sizes


def _max_pool_instance_probabilities(bag_slice: Array2D) -> Array1D:
    return np.max(bag_slice, axis=0)


def _mean_pool_instance_probabilities(bag_slice: Array2D) -> Array1D:
    return np.mean(bag_slice, axis=0)


def _logsumexp_pool_instance_probabilities(bag_slice: Array2D) -> Array1D:
    max_vals = np.max(bag_slice, axis=0, keepdims=True)
    pooled = np.log(np.sum(np.exp(bag_slice - max_vals), axis=0)) + max_vals.ravel()
    return np.exp(pooled - np.max(pooled))


def _pool_bag_instance_probabilities(bag_slice: Array2D, mil_cfg: MILConfig) -> Array1D:
    if mil_cfg.pooling == "max":
        return _max_pool_instance_probabilities(bag_slice)
    if mil_cfg.pooling == "mean":
        return _mean_pool_instance_probabilities(bag_slice)
    if mil_cfg.pooling == "logsumexp":
        return _logsumexp_pool_instance_probabilities(bag_slice)
    raise ValueError(f"Unsupported MIL pooling method: {mil_cfg.pooling}")


def pool_instance_probabilities(
    instance_proba: Any,
    bag_sizes: Any,
    mil_cfg: MILConfig,
) -> Array2D:
    """Pool instance probabilities into bag probabilities."""
    proba = np.asarray(instance_proba, dtype=float)
    sizes = np.asarray(bag_sizes, dtype=int)
    if proba.ndim != 2:
        raise ValueError("instance_proba must be a 2D array")
    if sizes.ndim != 1:
        raise ValueError("bag_sizes must be a 1D array")
    if sizes.sum() != proba.shape[0]:
        raise ValueError(
            "bag_sizes must sum to the number of instance predictions "
            f"({sizes.sum()} != {proba.shape[0]})"
        )

    pooled_rows: list[np.ndarray] = []
    start = 0
    for bag_size in sizes:
        bag_slice = proba[start:start + int(bag_size)]
        start += int(bag_size)

        pooled = np.asarray(_pool_bag_instance_probabilities(bag_slice, mil_cfg), dtype=float)
        denom = pooled.sum()
        pooled_rows.append(pooled / denom if denom > 0.0 else np.full_like(pooled, 1.0 / len(pooled)))

    return np.vstack(pooled_rows) if pooled_rows else np.empty((0, proba.shape[1]), dtype=float)


def predict_mil_proba(
    predict_instance_proba_fn: Callable[[Array2D], Array2D],
    x: Any,
    mil_cfg: MILConfig,
) -> Array2D:
    """Predict bag probabilities by flattening instances and pooling back."""
    x_flat, _, bag_sizes = flatten_mil_bags(x=x, y=None)
    if x_flat.shape[0] == 0:
        return np.empty((0, 0), dtype=float)
    instance_proba = np.asarray(predict_instance_proba_fn(x_flat), dtype=float)
    return pool_instance_probabilities(instance_proba=instance_proba, bag_sizes=bag_sizes, mil_cfg=mil_cfg)


def mil_multiclass_log_loss(
    y_true: Any,
    y_pred: Any,
    labels: Iterable[Any] | None = None,
) -> float:
    """Compute multiclass log loss on bag-level predictions."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(log_loss(y_true_arr, y_pred_arr, labels=None if labels is None else list(labels)))


def build_mil_feature_bags(
    chunks: Iterable[np.ndarray] | np.ndarray,
    pipeline_cfg: PipelineConfig,
    bag_ids: Any | None = None,
) -> MILFeatureBags:
    """Build one MIL bag per precomputed chunk or from a full audio waveform."""

    chunk_iterable: Iterable[np.ndarray]
    if isinstance(chunks, np.ndarray) and chunks.ndim == 1:
        chunk_iterable = split_audio_into_chunks(y=chunks, pipeline_cfg=pipeline_cfg)
    else:
        chunk_iterable = chunks

    bags: list[Array2D] = []
    feature_names_ref: list[str] | None = None

    for i, chunk in enumerate(chunk_iterable):
        bag, feature_names = build_mil_feature_matrix(
            y=np.asarray(chunk, dtype=float),
            pipeline_cfg=pipeline_cfg,
        )
        if feature_names_ref is None:
            feature_names_ref = list(feature_names)
        elif feature_names != feature_names_ref:
            raise RuntimeError("Inconsistent MIL feature names across bags")
        bags.append(bag)

    return MILFeatureBags(
        bags=bags,
        feature_names=feature_names_ref or [],
        bag_ids=None if bag_ids is None else np.asarray(bag_ids),
    )


def build_mil_bags_from_df(
    df,
    *,
    pipeline_cfg: PipelineConfig,
    indices: list[int] | None = None,
    pathroot: Path,
    filename_col: str = "filename",
    features_pathroot: Path | None = None,
) -> MILFeatureBags | tuple[list[Array2D], Array1D | dict[str, Array1D], list[str]]:
    """Load audio rows, split them into chunks, then build one MIL bag per chunk."""
    if indices is None:
        indices = list(range(len(df)))

    if features_pathroot is None:
        chunk_audio: list[np.ndarray] = []
        bag_ids: list[tuple[int, int]] = []
        expanded_indices: list[Any] = []
        for idx in indices:
            path = build_audio_path(
                df=df,
                idx=idx,
                pathroot=pathroot,
                filename_col=filename_col,
            )
            y = load_audio(path, sr=pipeline_cfg.feature.sr)
            chunks = split_audio_into_chunks(y=y, pipeline_cfg=pipeline_cfg)
            chunk_audio.extend(chunks)
            bag_ids.extend((idx, chunk_idx) for chunk_idx in range(len(chunks)))
            expanded_indices.extend([df.index[idx]] * len(chunks))

        bags = build_mil_feature_bags(
            chunks=chunk_audio,
            pipeline_cfg=pipeline_cfg,
            bag_ids=np.asarray(bag_ids, dtype=object),
        )
        return bags.bags, np.asarray(expanded_indices), bags.feature_names

    # Precomputed features mode: load one joblib per audio and split frame features into chunks.
    all_bags: list[Array2D] = []
    expanded_indices: list[Any] = []
    feature_names_ref: list[str] | None = None

    frame_rate_hz = float(pipeline_cfg.feature.sr) / float(pipeline_cfg.feature.hop_length)
    for idx in indices:
        filename = Path(df[filename_col].iloc[idx]).with_suffix(".joblib")
        feature_path = features_pathroot / filename
        frame_features = joblib_to_frame_features(feature_path)

        # Use only base (non-waveform, non-delta) features for frame-based chunking.
        frame_feature_items = [
            (k, arr)
            for k, v in frame_features.items()
            if k != "waveform" and not (k.endswith("_delta") or k.endswith("_delta2"))
            for arr in (np.asarray(v),)
        ]
        if not frame_feature_items:
            continue

        n_frames = min(
            v.shape[-1]
            for _, v in frame_feature_items
            if v.ndim >= 1
        )
        chunk_intervals = get_sliding_window_intervals(
            n_frames=n_frames,
            window_size_s=pipeline_cfg.chunk.chunk_size_s,
            step_size_s=pipeline_cfg.chunk.step_size_s,
            frame_rate_hz=frame_rate_hz,
        )

        for chunk_idx, (start, end) in enumerate(chunk_intervals):
            if end <= start:
                continue
            chunk_features = {
                key: value[..., start:end]
                for key, value in frame_feature_items
            }
            bag, feature_names = build_mil_feature_matrix(
                y=None,
                pipeline_cfg=pipeline_cfg,
                frame_features=chunk_features,
            )

            if feature_names_ref is None:
                feature_names_ref = list(feature_names)
            elif feature_names != feature_names_ref:
                raise RuntimeError("Inconsistent MIL feature names across precomputed chunks")

            all_bags.append(np.asarray(bag, dtype=float))
            expanded_indices.append(df.index[idx])
    return all_bags, np.asarray(expanded_indices), (feature_names_ref or [])
