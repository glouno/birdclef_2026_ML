from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import log_loss

from birdclef_2026_ml.feature_engineering.configs import (
    ChunkConfig,
    FeatureConfig,
    MILConfig,
    PoolingConfig,
)
from birdclef_2026_ml.feature_engineering.feature_extract import extract_all_frame_features
from birdclef_2026_ml.feature_engineering.pooling import global_pool
from birdclef_2026_ml.feature_engineering.utils import pad_short_audio
from birdclef_2026_ml.feature_engineering.chunking import get_chunk_intervals, feature_frame_rate_hz
from birdclef_2026_ml.models.mil_pooling import mil_pooling

Array1D = np.ndarray
Array2D = np.ndarray
MILBags = list[Array2D]


@dataclass(frozen=True)
class MILFeatureBags:
    """MIL-ready features grouped as chunk bags of pooled window features."""

    bags: MILBags
    feature_names: list[str]


def _validate_mil_enabled(mil_mode: bool, mil_config: MILConfig | None) -> None:
    if mil_mode and mil_config is None:
        raise ValueError("mil_config must be provided when mil_mode=True")


# def _window_intervals(duration_s: float, mil_cfg: MILConfig) -> list[tuple[float, float]]:
#     if duration_s <= 0.0:
#         return []

#     if duration_s <= mil_cfg.window_size_s:
#         return [(0.0, duration_s)]

#     windows: list[tuple[float, float]] = []
#     start_s = 0.0
#     while start_s < duration_s:
#         end_s = min(start_s + mil_cfg.window_size_s, duration_s)
#         if end_s <= start_s:
#             break
#         windows.append((start_s, end_s))
#         if end_s >= duration_s:
#             break
#         start_s += mil_cfg.step_size_s
#     return windows


# def _slice_feature_by_time(
#     feature_matrix: np.ndarray,
#     start_s: float,
#     end_s: float,
#     frame_rate_hz: float,
# ) -> np.ndarray:
#     x = np.asarray(feature_matrix, dtype=float)
#     if x.ndim == 1:
#         x = x[np.newaxis, :]
#     if x.ndim != 2:
#         raise ValueError("feature_matrix must be 1D or 2D")

#     n_frames = x.shape[1]
#     if n_frames == 0:
#         return np.empty((x.shape[0], 0), dtype=float)

#     start_idx = int(np.floor(start_s * frame_rate_hz))
#     end_idx = int(np.ceil(end_s * frame_rate_hz))

#     start_idx = min(max(start_idx, 0), n_frames - 1)
#     end_idx = min(max(end_idx, start_idx + 1), n_frames)
#     return x[:, start_idx:end_idx]


# def _feature_frame_rate_hz(feature_name: str, feature_cfg: FeatureConfig) -> float:
#     if feature_name == "waveform":
#         return float(feature_cfg.sr)
#     return float(feature_cfg.sr) / float(feature_cfg.hop_length)


def build_mil_feature_bags(
    y: np.ndarray,
    feature_cfg: FeatureConfig,
    pooling_cfg: PoolingConfig,
    chunk_cfg: ChunkConfig,
    mil_cfg: MILConfig,
) -> MILFeatureBags:
    """Build chunk bags where each row is a pooled MIL window feature vector."""
    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.size == 0:
        raise ValueError("y must contain at least one sample")

    # min_samples = int(np.ceil(float(chunk_cfg.chunks_s) * float(feature_cfg.sr)))
    # y_arr = _pad_short_audio_randomly(y_arr, min_samples=min_samples)
    y_arr = pad_short_audio(y_arr, chunk_cfg, feature_cfg)

    frame_features = extract_all_frame_features(y_arr, feature_cfg)
    audio_duration_s = float(y_arr.size) / float(feature_cfg.sr)

    n_frames = y.shape[1]
    if n_frames == 0:
        return np.array([], dtype=float)

    chunk_intervals = get_chunk_intervals(audio_duration_s, chunk_cfg.chunks_s, chunk_cfg.overlap, frame_rate_hz)

    if not chunk_intervals:
        return MILFeatureBags(bags=[], feature_names=[])

    per_feature_names: list[list[str]] = []
    for feature_name, feature_matrix in frame_features.items():
        pooled_template = global_pool(feature_matrix, pooling_cfg)
        per_feature_names.append(
            [f"{feature_name}_{idx:04d}" for idx in range(1, pooled_template.size + 1)]
        )

    feature_names = [name for names in per_feature_names for name in names]
    bags: MILBags = []

    for chunk_start_s, chunk_end_s in chunk_intervals:
        chunk_duration_s = chunk_end_s - chunk_start_s
        windows = _window_intervals(chunk_duration_s, mil_cfg)
        if not windows:
            continue

        window_rows: list[np.ndarray] = []
        for window_start_s, window_end_s in windows:
            row_parts: list[np.ndarray] = []
            abs_start_s = chunk_start_s + window_start_s
            abs_end_s = chunk_start_s + window_end_s
            for feature_name, feature_matrix in frame_features.items():
                frame_rate_hz = _feature_frame_rate_hz(feature_name, feature_cfg)
                segment = _slice_feature_by_time(
                    feature_matrix=feature_matrix,
                    start_s=abs_start_s,
                    end_s=abs_end_s,
                    frame_rate_hz=frame_rate_hz,
                )
                row_parts.append(global_pool(segment, pooling_cfg))
            window_rows.append(np.concatenate(row_parts))

        bags.append(np.vstack(window_rows))

    return MILFeatureBags(bags=bags, feature_names=feature_names)


def build_mil_bags_from_df(
    df: Any,
    label_cols: str | list[str],
    feature_cfg: FeatureConfig,
    pooling_cfg: PoolingConfig,
    chunk_cfg: ChunkConfig,
    mil_cfg: MILConfig,
    indices: list[int] | None = None,
    pathroot: str = "train_audio_dir",
    filename_col: str = "filename",
) -> tuple[MILBags, Array1D | dict[str, Array1D], list[str]]:
    """Build MIL bags from dataframe rows, expanding labels per produced chunk."""
    from birdclef_2026_ml.processing.audio_utils import build_audio_path, load_audio

    if isinstance(label_cols, str):
        label_names = [label_cols]
        return_single = True
    else:
        label_names = list(label_cols)
        return_single = False

    if indices is None:
        indices = list(range(len(df)))

    bags: MILBags = []
    labels_expanded: dict[str, list[Any]] = {label: [] for label in label_names}
    feature_names_ref: list[str] | None = None

    for idx in indices:
        audio_path = build_audio_path(
            df=df,
            idx=idx,
            pathroot=pathroot,
            filename_col=filename_col,
        )
        y = load_audio(audio_path, sr=feature_cfg.sr)
        bagged = build_mil_feature_bags(
            y=y,
            feature_cfg=feature_cfg,
            pooling_cfg=pooling_cfg,
            chunk_cfg=chunk_cfg,
            mil_cfg=mil_cfg,
        )

        if feature_names_ref is None:
            feature_names_ref = bagged.feature_names
        elif feature_names_ref != bagged.feature_names:
            raise RuntimeError("Inconsistent MIL feature schema across rows")

        row_labels = df.iloc[idx]
        for bag in bagged.bags:
            bags.append(np.asarray(bag, dtype=float))
            for label in label_names:
                labels_expanded[label].append(row_labels[label])

    labels_out = {label: np.asarray(values) for label, values in labels_expanded.items()}
    if return_single:
        return bags, labels_out[label_names[0]], (feature_names_ref or [])
    return bags, labels_out, (feature_names_ref or [])


def _ensure_2d_bag(bag: np.ndarray, bag_idx: int) -> Array2D:
    arr = np.asarray(bag, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Each MIL bag must be 2D, got shape={arr.shape} for bag {bag_idx}")
    if arr.shape[0] == 0:
        raise ValueError(f"MIL bag {bag_idx} contains no windows")
    return arr


def validate_mil_bags(x: Sequence[np.ndarray]) -> MILBags:
    bags = [_ensure_2d_bag(bag, bag_idx=i) for i, bag in enumerate(x)]
    if not bags:
        raise ValueError("MIL mode requires at least one bag")

    n_features = bags[0].shape[1]
    for i, bag in enumerate(bags[1:], start=1):
        if bag.shape[1] != n_features:
            raise ValueError(
                "All MIL bags must have the same number of features, "
                f"got {bag.shape[1]} and expected {n_features} at bag {i}"
            )
    return bags


def flatten_mil_bags(
    x: Sequence[np.ndarray],
    y: Sequence[Any] | np.ndarray | None = None,
) -> tuple[Array2D, Array1D | None, Array1D]:
    bags = validate_mil_bags(x)
    bag_sizes = np.asarray([bag.shape[0] for bag in bags], dtype=int)
    x_flat = np.vstack(bags)

    if y is None:
        return x_flat, None, bag_sizes

    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError("MIL labels must be 1D")
    if y_arr.shape[0] != len(bags):
        raise ValueError(
            "MIL labels must have one value per bag, "
            f"got {y_arr.shape[0]} labels for {len(bags)} bags"
        )
    y_flat = np.repeat(y_arr, bag_sizes)
    return x_flat, y_flat, bag_sizes


def pool_instance_probabilities(
    probs: Array2D,
    bag_sizes: Sequence[int],
    mil_cfg: MILConfig,
) -> Array2D:
    """Pool instance-level probabilities back to chunk-level probabilities."""
    probs_arr = np.asarray(probs, dtype=float)
    bag_sizes_arr = np.asarray(bag_sizes, dtype=int)

    if probs_arr.ndim != 2:
        raise ValueError("probs must be a 2D array")

    if probs_arr.shape[0] != bag_sizes_arr.sum():
        raise ValueError(
            "Probability rows and MIL bag sizes mismatch "
            f"({probs_arr.shape[0]} != {bag_sizes_arr.sum()})"
        )

    pooled: list[np.ndarray] = []
    start = 0
    for bag_size in bag_sizes_arr:
        stop = start + int(bag_size)
        pooled.append(mil_pooling(probs_arr[start:stop], method=mil_cfg.pooling))
        start = stop
    return np.vstack(pooled)


def predict_mil_proba(
    predict_instance_proba_fn: Any,
    x: Sequence[np.ndarray],
    mil_cfg: MILConfig,
) -> Array2D:
    """Run instance-level prediction then MIL-pool back to one row per bag."""
    x_flat, _, bag_sizes = flatten_mil_bags(x=x, y=None)
    instance_probs = np.asarray(predict_instance_proba_fn(x_flat), dtype=float)
    return pool_instance_probabilities(instance_probs, bag_sizes=bag_sizes, mil_cfg=mil_cfg)


def mil_multiclass_log_loss(
    y_true: Sequence[Any] | np.ndarray,
    pooled_probs: Array2D,
    classes: Sequence[Any] | np.ndarray,
) -> float:
    """Compute final chunk-level multiclass log loss after MIL pooling."""
    y_true_arr = np.asarray(y_true)
    probs_arr = np.asarray(pooled_probs, dtype=float)
    classes_arr = np.asarray(classes)
    if y_true_arr.shape[0] != probs_arr.shape[0]:
        raise ValueError(
            "MIL pooled probabilities and labels must align, "
            f"got {probs_arr.shape[0]} rows and {y_true_arr.shape[0]} labels"
        )
    return float(log_loss(y_true_arr, probs_arr, labels=classes_arr))
