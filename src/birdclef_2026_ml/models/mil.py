from typing import Iterable

import numpy as np

from birdclef_2026_ml.configs import PipelineConfig, load_pipeline_config


def _estimate_windows_per_chunk(config: PipelineConfig) -> int:
    chunk_seconds = float(config.chunk.chunk_size_s)
    window_seconds = float(config.mil.chunk_size_s)
    step_seconds = float(config.mil.step_size_s)

    windows = np.floor((chunk_seconds - window_seconds) / step_seconds) + 1
    return max(1, int(windows))


def assign_bag_ids(window_ids: Iterable[int], run_name: str) -> np.ndarray:
    """Assign MIL bag ids with fixed bag size and enforced full last bag via backward overlap."""
    config = load_pipeline_config(run_name)
    K = _estimate_windows_per_chunk(config)

    window_ids = np.asarray(list(window_ids), dtype=int).reshape(-1)
    n = len(window_ids)

    if n == 0:
        return np.empty(0, dtype=int)

    bag_ids = np.empty(n, dtype=int)
    if n <= K:
        bag_ids[:] = 0
        return bag_ids

    # number of full bags
    n_full = n // K
    remainder = n % K

    # if perfectly divisible -> standard grouping
    if remainder == 0:
        for i in range(n):
            bag_ids[i] = i // K
        return bag_ids

    # last bag must be full -> shift start backward
    n_bags = n_full + 1

    # assign all full bags normally
    for i in range(n_full * K):
        bag_ids[i] = i // K

    # build last bag with overlap
    start_last = n - K

    for i in range(start_last, n):
        bag_ids[i] = n_bags - 1

    return bag_ids


def bag_order(bag_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bag_ids = np.asarray(bag_ids, dtype=int).reshape(-1)
    if bag_ids.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    uniq_ids, first_idx = np.unique(bag_ids, return_index=True)
    order = np.argsort(first_idx)
    return uniq_ids[order], first_idx[order]


def collapse_bag_labels(bag_ids: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Collapse labels so each bag id maps to a single label value."""
    bag_ids = np.asarray(bag_ids, dtype=int).reshape(-1)
    y = np.asarray(y)
    if bag_ids.shape[0] != y.shape[0]:
        raise ValueError("y must have same length as bag_ids")
    _, first_idx = bag_order(bag_ids)
    return y[first_idx]


def collapse_bag_multilabel_targets(bag_ids: np.ndarray, y: np.ndarray) -> np.ndarray:
    bag_ids = np.asarray(bag_ids, dtype=int).reshape(-1)
    y = np.asarray(y, dtype=np.int8)
    if y.ndim != 2:
        raise ValueError("y must be a 2D multilabel matrix")
    if bag_ids.shape[0] != y.shape[0]:
        raise ValueError("y must have same length as bag_ids")

    uniq_ids, _ = bag_order(bag_ids)
    collapsed = np.zeros((uniq_ids.shape[0], y.shape[1]), dtype=np.int8)
    for i, bag_id in enumerate(uniq_ids):
        collapsed[i] = np.max(y[bag_ids == bag_id], axis=0)
    return collapsed
