from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from birdclef_2026_ml.configs import PipelineConfig, load_pipeline_config


# BAG_STAT_NAMES: tuple[str, ...] = (
#     "mean",
#     "std",
#     "max",
#     "top2_median",
#     "top3_median",
#     "noisy_or",
#     "entropy",
# )


# @dataclass(frozen=True)
# class ProbabilityBag:
#     bag_id: int
#     primary_label_proba: np.ndarray
#     class_name_proba: np.ndarray
#     primary_target: np.ndarray
#     source_name: str

# # ----- Everything related to data augmentation -----


# def build_bag_feature_names(class_labels: Iterable[object], prefix: str) -> list[str]:
#     labels = [str(label) for label in class_labels]
#     return [
#         f"{prefix}__{label}__{stat_name}"
#         for stat_name in BAG_STAT_NAMES
#         for label in labels
#     ]


# def build_dual_head_bag_feature_matrix(
#     primary_label_proba: np.ndarray,
#     class_name_proba: np.ndarray,
#     bag_ids: np.ndarray,
# ) -> np.ndarray:
#     primary_features = build_bag_feature_matrix(primary_label_proba, bag_ids)
#     class_features = build_bag_feature_matrix(class_name_proba, bag_ids)
#     return np.concatenate([primary_features, class_features], axis=1)


# def build_dual_head_bag_feature_names(
#     primary_label_labels: Iterable[object],
#     class_name_labels: Iterable[object],
# ) -> list[str]:
#     return build_bag_feature_names(primary_label_labels, "primary_label") + build_bag_feature_names(
#         class_name_labels,
#         "class_name",
#     )


# def collapse_bag_multilabel_targets(bag_ids: np.ndarray, y: np.ndarray) -> np.ndarray:
#     bag_ids = np.asarray(bag_ids, dtype=int).reshape(-1)
#     y = np.asarray(y, dtype=np.int8)
#     if y.ndim != 2:
#         raise ValueError("y must be a 2D multilabel matrix")
#     if bag_ids.shape[0] != y.shape[0]:
#         raise ValueError("y must have same length as bag_ids")

#     uniq_ids, _ = bag_order(bag_ids)
#     collapsed = np.zeros((uniq_ids.shape[0], y.shape[1]), dtype=np.int8)
#     for i, bag_id in enumerate(uniq_ids):
#         collapsed[i] = np.max(y[bag_ids == bag_id], axis=0)
#     return collapsed


# def split_probabilities_by_bag(
#     proba: np.ndarray,
#     bag_ids: np.ndarray,
# ) -> list[np.ndarray]:
#     proba = np.asarray(proba, dtype=float)
#     bag_ids = np.asarray(bag_ids, dtype=int).reshape(-1)
#     if proba.ndim != 2:
#         raise ValueError("proba must have shape (n_samples, n_classes)")
#     if bag_ids.shape[0] != proba.shape[0]:
#         raise ValueError("bag_ids must have same length as proba")

#     uniq_ids, _ = bag_order(bag_ids)
#     return [np.asarray(proba[bag_ids == bag_id], dtype=float) for bag_id in uniq_ids]


# def build_probability_bags(
#     primary_label_proba: np.ndarray,
#     class_name_proba: np.ndarray,
#     bag_ids: np.ndarray,
#     primary_targets: np.ndarray,
#     *,
#     source_names: Iterable[object] | None = None,
# ) -> list[ProbabilityBag]:
#     primary_sequences = split_probabilities_by_bag(primary_label_proba, bag_ids)
#     class_sequences = split_probabilities_by_bag(class_name_proba, bag_ids)
#     targets = np.asarray(primary_targets)
#     if targets.ndim == 1:
#         raise ValueError("primary_targets must already be bag-level one-hot or multi-hot")
#     if targets.shape[0] != len(primary_sequences):
#         raise ValueError("primary_targets must have same bag count as grouped probabilities")

#     if source_names is None:
#         names = [f"bag_{bag_id}" for bag_id in bag_order(np.asarray(bag_ids, dtype=int))[0]]
#     else:
#         names = [str(name) for name in source_names]
#         if len(names) != len(primary_sequences):
#             raise ValueError("source_names must match bag count")

#     bags: list[ProbabilityBag] = []
#     for bag_idx, (primary_seq, class_seq, target, source_name) in enumerate(
#         zip(primary_sequences, class_sequences, targets, names)
#     ):
#         bags.append(
#             ProbabilityBag(
#                 bag_id=bag_idx,
#                 primary_label_proba=np.asarray(primary_seq, dtype=float),
#                 class_name_proba=np.asarray(class_seq, dtype=float),
#                 primary_target=np.asarray(target, dtype=np.int8),
#                 source_name=source_name,
#             )
#         )
#     return bags


# def estimate_mixture_size_distribution(
#     y_multilabel: np.ndarray,
#     *,
#     max_mixture_size: int,
# ) -> dict[int, float]:
#     y_multilabel = np.asarray(y_multilabel, dtype=np.int8)
#     if y_multilabel.ndim != 2:
#         raise ValueError("y_multilabel must be 2D")

#     counts = np.clip(np.sum(y_multilabel, axis=1), 1, max_mixture_size).astype(int)
#     if counts.size == 0:
#         return {1: 1.0}

#     uniq, freq = np.unique(counts, return_counts=True)
#     probs = freq.astype(float) / float(freq.sum())
#     return {int(k): float(v) for k, v in zip(uniq, probs)}


# def normalize_mixture_size_distribution(
#     probabilities: dict[int, float],
#     *,
#     max_mixture_size: int,
# ) -> tuple[np.ndarray, np.ndarray]:
#     if not probabilities:
#         support = np.arange(1, max_mixture_size + 1, dtype=int)
#         probs = np.zeros_like(support, dtype=float)
#         probs[0] = 1.0
#         return support, probs

#     support = np.asarray(
#         sorted(
#             {
#                 int(np.clip(key, 1, max_mixture_size))
#                 for key, value in probabilities.items()
#                 if float(value) > 0.0
#             }
#         ),
#         dtype=int,
#     )
#     if support.size == 0:
#         return np.array([1], dtype=int), np.array([1.0], dtype=float)

#     probs = np.asarray(
#         [max(0.0, float(probabilities[int(key)])) for key in support],
#         dtype=float,
#     )
#     total = float(probs.sum())
#     if total <= 0.0:
#         probs = np.zeros_like(probs)
#         probs[0] = 1.0
#         return support, probs
#     return support, probs / total


# def align_probability_sequence(
#     sequence: np.ndarray,
#     target_length: int,
#     *,
#     fill_value: np.ndarray,
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     sequence = np.asarray(sequence, dtype=float)
#     if sequence.ndim != 2:
#         raise ValueError("sequence must be 2D")
#     if sequence.shape[0] == target_length:
#         return sequence.copy()
#     if sequence.shape[0] > target_length:
#         start = int(rng.integers(0, sequence.shape[0] - target_length + 1))
#         return np.asarray(sequence[start:start + target_length], dtype=float)

#     out = np.repeat(np.asarray(fill_value, dtype=float)[np.newaxis, :], target_length, axis=0)
#     max_offset = target_length - sequence.shape[0]
#     offset = int(rng.integers(0, max_offset + 1))
#     out[offset:offset + sequence.shape[0]] = sequence
#     return out


# def mix_probability_sequences(
#     sequences: list[np.ndarray],
#     *,
#     rng: np.random.Generator,
#     method: str,
#     weight_min: float,
#     weight_max: float,
#     fill_value: np.ndarray,
#     target_length: int | None = None,
# ) -> tuple[np.ndarray, np.ndarray]:
#     if not sequences:
#         raise ValueError("At least one sequence is required")

#     lengths = np.asarray([seq.shape[0] for seq in sequences], dtype=int)
#     if target_length is None:
#         target_length = int(rng.choice(lengths))

#     aligned = np.stack(
#         [
#             align_probability_sequence(
#                 seq,
#                 target_length,
#                 fill_value=fill_value,
#                 rng=rng,
#             )
#             for seq in sequences
#         ],
#         axis=0,
#     )

#     if method == "max":
#         weights = np.full(aligned.shape[0], 1.0 / aligned.shape[0], dtype=float)
#         return np.max(aligned, axis=0), weights
#     if method != "weighted":
#         raise ValueError(f"Unsupported mixing method: {method}")

#     weights = rng.uniform(weight_min, weight_max, size=aligned.shape[0]).astype(float)
#     weights /= weights.sum()
#     mixed = np.tensordot(weights, aligned, axes=(0, 0))
#     return np.asarray(mixed, dtype=float), weights


# def apply_temporal_mask(
#     sequence: np.ndarray,
#     *,
#     rng: np.random.Generator,
#     keep_ratio_min: float,
#     keep_ratio_max: float,
#     min_segments: int,
#     max_segments: int,
#     fill_value: np.ndarray,
# ) -> np.ndarray:
#     sequence = np.asarray(sequence, dtype=float)
#     if sequence.ndim != 2:
#         raise ValueError("sequence must be 2D")
#     if sequence.shape[0] <= 1:
#         return sequence.copy()

#     mask = sample_temporal_mask(
#         sequence.shape[0],
#         rng=rng,
#         keep_ratio_min=keep_ratio_min,
#         keep_ratio_max=keep_ratio_max,
#         min_segments=min_segments,
#         max_segments=max_segments,
#     )
#     out = sequence.copy()
#     out[mask] = np.asarray(fill_value, dtype=float)
#     return out


# def sample_temporal_mask(
#     sequence_length: int,
#     *,
#     rng: np.random.Generator,
#     keep_ratio_min: float,
#     keep_ratio_max: float,
#     min_segments: int,
#     max_segments: int,
# ) -> np.ndarray:
#     if sequence_length <= 1:
#         return np.zeros(sequence_length, dtype=bool)

#     target_keep_ratio = float(rng.uniform(keep_ratio_min, keep_ratio_max))
#     to_mask = int(round(sequence_length * (1.0 - target_keep_ratio)))
#     if to_mask <= 0:
#         return np.zeros(sequence_length, dtype=bool)

#     n_segments = int(rng.integers(min_segments, max_segments + 1))
#     remaining = min(to_mask, sequence_length)
#     mask = np.zeros(sequence_length, dtype=bool)

#     for segment_idx in range(n_segments):
#         segments_left = n_segments - segment_idx
#         if remaining <= 0:
#             break
#         max_len = max(1, remaining - (segments_left - 1))
#         segment_len = int(rng.integers(1, max_len + 1))
#         start = int(rng.integers(0, max(1, sequence_length - segment_len + 1)))
#         mask[start:start + segment_len] = True
#         remaining -= segment_len

#     return mask


# def apply_probability_noise(
#     sequence: np.ndarray,
#     *,
#     sigma: float,
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     sequence = np.asarray(sequence, dtype=float)
#     if sigma <= 0:
#         return np.clip(sequence, 0.0, 1.0)
#     noise = rng.normal(loc=0.0, scale=sigma, size=sequence.shape)
#     return np.clip(sequence + noise, 0.0, 1.0)
