from __future__ import annotations

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
    """Assign bag ids for MIL windows using the run pipeline config."""
    config = load_pipeline_config(run_name)
    windows_per_chunk = _estimate_windows_per_chunk(config)
    window_ids = np.asarray(list(window_ids), dtype=int).reshape(-1)
    if window_ids.size == 0:
        return np.empty(0, dtype=int)

    bag_ids = np.empty_like(window_ids, dtype=int)
    bag_id = 0
    count_in_bag = 0

    for idx, window_id in enumerate(window_ids):
        if count_in_bag > 0 and (window_id == 0 or count_in_bag >= windows_per_chunk):
            bag_id += 1
            count_in_bag = 0
        bag_ids[idx] = bag_id
        count_in_bag += 1

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


def build_bag_feature_matrix(
    proba: np.ndarray,
    bag_ids: np.ndarray,
    *,
    bag_batch_size: int | None = None,
) -> np.ndarray:
    """Aggregate window-level probabilities into bag-level features."""
    proba = np.asarray(proba, dtype=float)
    bag_ids = np.asarray(bag_ids, dtype=int).reshape(-1)
    if proba.ndim != 2:
        raise ValueError("proba must have shape (n_samples, n_classes)")
    if bag_ids.shape[0] != proba.shape[0]:
        raise ValueError("bag_ids must have same length as proba")

    ordered_bag_ids, _ = bag_order(bag_ids)

    n_bags = ordered_bag_ids.shape[0]
    n_classes = proba.shape[1]
    n_stats = 12
    features = np.zeros((n_bags, n_classes * n_stats), dtype=float)

    if bag_batch_size is None or bag_batch_size <= 0:
        bag_batch_size = n_bags

    for start in range(0, n_bags, bag_batch_size):
        stop = min(start + bag_batch_size, n_bags)
        for offset, bag_id in enumerate(ordered_bag_ids[start:stop]):
            i = start + offset
            mask = bag_ids == bag_id
            bag_proba = proba[mask]
            if bag_proba.shape[0] == 0:
                continue

            mean = np.mean(bag_proba, axis=0)
            std = np.std(bag_proba, axis=0)
            maxv = np.max(bag_proba, axis=0)

            if bag_proba.shape[0] >= 2:
                top2 = np.partition(bag_proba, -2, axis=0)[-2:, :]
                top2_med = np.median(top2, axis=0)
            else:
                top2_med = np.median(bag_proba, axis=0)

            if bag_proba.shape[0] >= 3:
                top3 = np.partition(bag_proba, -3, axis=0)[-3:, :]
                top3_med = np.median(top3, axis=0)
            else:
                top3_med = np.median(bag_proba, axis=0)

            if bag_proba.shape[0] >= 5:
                top5 = np.partition(bag_proba, -5, axis=0)[-5:, :]
                top5_med = np.median(top5, axis=0)
            else:
                top5_med = np.median(bag_proba, axis=0)

            p90 = np.percentile(bag_proba, 90, axis=0)
            p80 = np.percentile(bag_proba, 80, axis=0)
            above_p80 = bag_proba > p80
            count_above_p80 = np.sum(above_p80, axis=0).astype(float)

            longest_run = np.zeros(n_classes, dtype=float)
            spread_active = np.zeros(n_classes, dtype=float)
            for c in range(n_classes):
                active = above_p80[:, c]
                if not np.any(active):
                    continue
                run = 0
                best = 0
                for flag in active:
                    if flag:
                        run += 1
                        if run > best:
                            best = run
                    else:
                        run = 0
                longest_run[c] = float(best)
                positions = np.flatnonzero(active).astype(float)
                spread_active[c] = float(np.var(positions)) if positions.size > 0 else 0.0

            noisy_or = 1.0 - np.prod(1.0 - bag_proba, axis=0)

            prob_sum = np.sum(bag_proba, axis=0)
            prob_norm = np.divide(
                bag_proba,
                prob_sum,
                out=np.zeros_like(bag_proba),
                where=prob_sum > 0,
            )
            log_prob = np.log(np.clip(prob_norm, 1e-12, 1.0))
            entropy = -np.sum(prob_norm * log_prob, axis=0)

            features[i] = np.concatenate(
                [
                    mean,
                    std,
                    maxv,
                    top2_med,
                    top3_med,
                    top5_med,
                    p90,
                    count_above_p80,
                    longest_run,
                    spread_active,
                    noisy_or,
                    entropy,
                ],
                axis=0,
            )

    return features
