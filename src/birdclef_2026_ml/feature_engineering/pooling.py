
import numpy as np
from typing import Literal

from birdclef_2026_ml.configs import PipelineConfig, PoolingConfig
from birdclef_2026_ml.feature_engineering.chunking import (
    feature_frame_rate_hz,
    get_sliding_window_intervals,
)


def _nan_skew(x: np.ndarray, axis: int = -1) -> np.ndarray:
    mean = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True)
    centered = x - mean
    m3 = np.nanmean(centered ** 3, axis=axis)
    s3 = np.squeeze(std, axis=axis) ** 3
    return np.where(s3 == 0.0, 0.0, m3 / s3)


def _nan_kurtosis(x: np.ndarray, axis: int = -1) -> np.ndarray:
    mean = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True)
    centered = x - mean
    m4 = np.nanmean(centered ** 4, axis=axis)
    s4 = np.squeeze(std, axis=axis) ** 4
    return np.where(s4 == 0.0, -3.0, m4 / s4 - 3.0)


def global_pool(feature_matrix: np.ndarray, pooling_cfg: PoolingConfig) -> np.ndarray:
    """Pool a feature matrix of shape (n_features, n_frames) into 1 vector."""
    x = np.asarray(feature_matrix, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2:
        raise ValueError("feature_matrix must be 1D or 2D")

    pieces: list[np.ndarray] = []
    for stat in pooling_cfg.stats:
        if stat == "mean":
            arr = np.nanmean(x, axis=1)
        elif stat == "std":
            arr = np.nanstd(x, axis=1)
        elif stat == "min":
            arr = np.nanmin(x, axis=1)
        elif stat == "max":
            arr = np.nanmax(x, axis=1)
        elif stat == "skew":
            arr = _nan_skew(x, axis=1)
        elif stat == "kurtosis":
            arr = _nan_kurtosis(x, axis=1)
        else:
            raise ValueError(f"Unsupported stat: {stat}")
        pieces.append(np.asarray(arr, dtype=float).ravel())

    if pooling_cfg.percentiles:
        p = np.asarray(pooling_cfg.percentiles, dtype=float)
        pct = np.nanpercentile(x, p, axis=1).T.reshape(-1)
        pieces.append(np.asarray(pct, dtype=float))

    if not pieces:
        return np.array([], dtype=float)

    out = np.concatenate(pieces)
    out = np.nan_to_num(out, nan=pooling_cfg.nan_fill_value)
    return out


def sliding_window_pool(
    feature_matrix: np.ndarray,
    pooling_cfg: PoolingConfig,
    window_size_s: float,
    step_size_s: float,
    frame_rate_hz: float,
) -> np.ndarray:
    """Pool features over arbitrary sliding windows and return one row per window."""
    x = np.asarray(feature_matrix, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2:
        raise ValueError("feature_matrix must be 1D or 2D")

    n_frames = x.shape[1]
    if n_frames == 0:
        return np.array([], dtype=float)

    window_indices = get_sliding_window_intervals(
        n_frames=n_frames,
        window_size_s=window_size_s,
        step_size_s=step_size_s,
        frame_rate_hz=frame_rate_hz,
    )

    windows: list[np.ndarray] = []
    for start, end in window_indices:
        if end <= start:
            continue
        windows.append(global_pool(x[:, start:end], pooling_cfg))

    if not windows:
        return np.empty((0, 0), dtype=float)
    return np.vstack(windows)


def pool_feature_dict(
    features_dict: dict[str, np.ndarray],
    pipeline_cfg: PipelineConfig
) -> dict[str, np.ndarray]:
    """Pool each feature matrix in a dict using global or chunk mode."""
    pooled: dict[str, np.ndarray] = {}

    if pipeline_cfg.pooling.pooling_mode == "global":
        for name, mat in features_dict.items():
            pooled[name] = global_pool(mat, pipeline_cfg.pooling)
        return pooled

    if pipeline_cfg.pooling.pooling_mode == "chunk":
        for name, mat in features_dict.items():
            frame_rate_hz = feature_frame_rate_hz(name, pipeline_cfg.feature.sr, pipeline_cfg.feature.hop_length)
            pooled[name] = sliding_window_pool(
                feature_matrix=mat,
                pooling_cfg=pipeline_cfg.pooling,
                window_size_s=pipeline_cfg.chunk.chunk_size_s,
                step_size_s=pipeline_cfg.chunk.step_size_s,
                frame_rate_hz=frame_rate_hz,
            )
        return pooled

    return pooled


def pool_feature_dict_sliding_windows(
    features_dict: dict[str, np.ndarray],
    pipeline_cfg: PipelineConfig,
) -> dict[str, np.ndarray]:
    """Pool each feature matrix into sliding-window instances for MIL-style inputs."""
    pooled: dict[str, np.ndarray] = {}

    for name, mat in features_dict.items():
        frame_rate_hz = feature_frame_rate_hz(name, pipeline_cfg.feature.sr, pipeline_cfg.feature.hop_length)
        pooled[name] = sliding_window_pool(
            feature_matrix=mat,
            pooling_cfg=pipeline_cfg.pooling,
            window_size_s=pipeline_cfg.mil.window_size_s,
            step_size_s=pipeline_cfg.mil.step_size_s,
            frame_rate_hz=frame_rate_hz,
        )
    return pooled
