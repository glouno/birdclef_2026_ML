"""
Multiple Instance Learning (MIL) pooling methods for audio event detection.
Splits 5-second windows into overlapping 1-second windows and pools probabilities.
"""
import numpy as np
from typing import Literal, Callable

PoolingMethod = Literal["max", "mean", "logsumexp"]


def split_into_overlapping_windows(
    arr: np.ndarray, window_size: int, step_size: int
) -> np.ndarray:
    """
    Split a 1D or 2D array into overlapping windows.
    Args:
        arr: shape (n_samples, n_features) or (n_samples,)
        window_size: number of frames per window
        step_size: stride between windows
    Returns:
        windows: shape (n_windows, window_size, ...)
    """
    arr = np.asarray(arr)
    n = arr.shape[0]
    n_windows = 1 + (n - window_size) // step_size
    return np.stack([
        arr[i * step_size: i * step_size + window_size]
        for i in range(n_windows)
    ])


def mil_pooling(
    probs: np.ndarray, method: PoolingMethod = "max"
) -> np.ndarray:
    """
    Pool probabilities across instances (windows) using MIL pooling.
    Args:
        probs: shape (n_windows, n_classes)
        method: pooling method ("max", "mean", "logsumexp")
    Returns:
        pooled: shape (n_classes,)
    """
    if method == "max":
        return np.max(probs, axis=0)
    elif method == "mean":
        return np.mean(probs, axis=0)
    elif method == "logsumexp":
        # log-sum-exp pooling (softmax-like)
        return np.exp(np.logaddexp.reduce(np.log(np.clip(probs, 1e-12, 1)), axis=0) - np.log(probs.shape[0]))
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def mil_predict(
    model: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    window_size: int = 1 * 50,  # e.g., 1s * 50 frames/sec
    step_size: int = 25,        # 0.5s overlap
    pooling: PoolingMethod = "max"
) -> np.ndarray:
    """
    Apply MIL prediction: split input into overlapping windows, predict, and pool.
    Args:
        model: function that takes (n_frames, n_features) and returns (n_windows, n_classes)
        x: shape (n_frames, n_features)
        window_size: number of frames per window (default: 1s)
        step_size: stride between windows (default: 0.5s overlap)
        pooling: pooling method
    Returns:
        pooled_probs: shape (n_classes,)
    """
    windows = split_into_overlapping_windows(x, window_size, step_size)
    probs = model(windows)  # shape (n_windows, n_classes)
    return mil_pooling(probs, method=pooling)
