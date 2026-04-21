from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.metrics import log_loss

from birdclef_2026_ml.feature_engineering import build_mil_feature_matrix, split_audio_into_chunks
from birdclef_2026_ml.feature_engineering.configs import ChunkConfig, FeatureConfig, MILConfig, PoolingConfig
from birdclef_2026_ml.processing.audio_utils import build_audio_path, load_audio


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
    feature_cfg: FeatureConfig,
    pooling_cfg: PoolingConfig,
    mil_cfg: MILConfig,
    chunk_cfg: ChunkConfig | None = None,
    bag_ids: Any | None = None,
) -> MILFeatureBags:
    """Build one MIL bag per precomputed chunk or from a full audio waveform."""
    chunk_iterable: Iterable[np.ndarray]
    if isinstance(chunks, np.ndarray) and chunks.ndim == 1:
        chunk_iterable = split_audio_into_chunks(
            y=chunks,
            feature_cfg=feature_cfg,
            chunk_cfg=chunk_cfg or ChunkConfig(),
        )
    else:
        chunk_iterable = chunks

    bags: list[Array2D] = []
    feature_names_ref: list[str] | None = None
    for chunk in chunk_iterable:
        bag, feature_names = build_mil_feature_matrix(
            y=np.asarray(chunk, dtype=float),
            feature_cfg=feature_cfg,
            pooling_cfg=pooling_cfg,
            mil_cfg=mil_cfg,
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
    feature_cfg: FeatureConfig,
    pooling_cfg: PoolingConfig,
    mil_cfg: MILConfig,
    label_cols: str | list[str] | None = None,
    chunk_cfg: ChunkConfig | None = None,
    indices: list[int] | None = None,
    pathroot: str = "train_audio_dir",
    filename_col: str = "filename",
) -> MILFeatureBags | tuple[list[Array2D], Array1D | dict[str, Array1D], list[str]]:
    """Load audio rows, split them into chunks, then build one MIL bag per chunk."""
    chunk_cfg = chunk_cfg or ChunkConfig()
    if indices is None:
        indices = list(range(len(df)))

    if label_cols is None:
        label_names: list[str] = []
        return_single = False
    elif isinstance(label_cols, str):
        label_names = [label_cols]
        return_single = True
    else:
        label_names = list(label_cols)
        return_single = False

    chunk_audio: list[np.ndarray] = []
    bag_ids: list[tuple[int, int]] = []
    labels_expanded: dict[str, list[Any]] = {label: [] for label in label_names}
    for idx in indices:
        path = build_audio_path(
            df=df,
            idx=idx,
            pathroot=pathroot,
            filename_col=filename_col,
        )
        y = load_audio(path, sr=feature_cfg.sr)
        chunks = split_audio_into_chunks(y=y, feature_cfg=feature_cfg, chunk_cfg=chunk_cfg)
        chunk_audio.extend(chunks)
        bag_ids.extend((idx, chunk_idx) for chunk_idx in range(len(chunks)))
        if label_names:
            row_labels = df.iloc[idx]
            for label in label_names:
                labels_expanded[label].extend([row_labels[label]] * len(chunks))

    bags = build_mil_feature_bags(
        chunks=chunk_audio,
        feature_cfg=feature_cfg,
        pooling_cfg=pooling_cfg,
        mil_cfg=mil_cfg,
        chunk_cfg=chunk_cfg,
        bag_ids=np.asarray(bag_ids, dtype=object),
    )
    if not label_names:
        return bags

    labels_out = {label: np.asarray(values) for label, values in labels_expanded.items()}
    if return_single:
        return bags.bags, labels_out[label_names[0]], bags.feature_names
    return bags.bags, labels_out, bags.feature_names
