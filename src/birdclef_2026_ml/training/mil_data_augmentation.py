# from __future__ import annotations

# from dataclasses import asdict
# from pathlib import Path

# import joblib
# import numpy as np
# import yaml
# from sklearn.multiclass import OneVsRestClassifier

# from birdclef_2026_ml.configs import (
#     artifact_stem,
#     build_second_stage_estimator,
#     load_experiment_config,
# )
# from birdclef_2026_ml.inference.ovr_inference import run_ovr_inference
# from birdclef_2026_ml.models.artifacts import (
#     DualOneVsRestArtifacts,
#     SecondStagePrimaryLabelArtifacts,
# )
# from birdclef_2026_ml.models.mil import (
#     ProbabilityBag,
#     align_probability_sequence,
#     apply_probability_noise,
#     assign_bag_ids,
#     build_dual_head_bag_feature_matrix,
#     build_dual_head_bag_feature_names,
#     build_probability_bags,
#     collapse_bag_labels,
#     collapse_bag_multilabel_targets,
#     estimate_mixture_size_distribution,
#     normalize_mixture_size_distribution,
#     sample_temporal_mask,
# )
# from birdclef_2026_ml.paths import load_project_paths
# from birdclef_2026_ml.processing.memmap_dataset import load_memmap_dataset


# DEFAULT_MIXTURE_SIZE_PROBABILITIES = {
#     1: 0.70,
#     2: 0.22,
#     3: 0.06,
#     4: 0.02,
# }


# def _resolve_source_model_path(
#     experiment_dir: Path,
#     artifact_stem_name: str,
#     model_filename: str | None,
# ) -> Path:
#     model_path = experiment_dir / (model_filename or f"{artifact_stem_name}_ovr.joblib")
#     if not model_path.exists():
#         raise FileNotFoundError(f"Saved OVR artifacts not found: {model_path}")
#     return model_path


# def _load_stage_one_artifacts(model_path: Path) -> DualOneVsRestArtifacts:
#     artifacts = joblib.load(model_path)
#     if not isinstance(artifacts, DualOneVsRestArtifacts):
#         raise TypeError("Stage-2 MIL training requires DualOneVsRestArtifacts")
#     return artifacts


# def _bag_ids_for_train_dataset(run_name: str, bags_meta: np.ndarray | None) -> np.ndarray:
#     if bags_meta is None:
#         raise ValueError("Train MIL second stage requires MIL bag metadata in train dataset")
#     bag_meta_arr = np.asarray(bags_meta)
#     if bag_meta_arr.ndim != 1:
#         raise ValueError("Train MIL bag metadata must be 1D window ids")
#     return assign_bag_ids(bag_meta_arr, run_name)


# def _bag_ids_for_soundscape_dataset(bags_meta: np.ndarray | None) -> np.ndarray:
#     if bags_meta is None:
#         raise ValueError("Soundscape MIL second stage requires MIL bag metadata")
#     bag_meta_arr = np.asarray(bags_meta)
#     if bag_meta_arr.ndim != 2 or bag_meta_arr.shape[1] < 1:
#         raise ValueError("Soundscape MIL bag metadata must have shape (n_rows, >=1)")
#     return np.asarray(bag_meta_arr[:, 0], dtype=int)


# def _single_label_to_multihot(y: np.ndarray, n_classes: int) -> np.ndarray:
#     y = np.asarray(y, dtype=int).reshape(-1)
#     out = np.zeros((y.shape[0], n_classes), dtype=np.int8)
#     out[np.arange(y.shape[0]), y] = 1
#     return out


# def _predict_head_probabilities(
#     run_name: str,
#     artifacts: DualOneVsRestArtifacts,
#     *,
#     soundscape: bool,
#     reduced: bool,
# ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray | None]:
#     dataset = load_memmap_dataset(run_name, reduced=reduced, soundscape=soundscape)
#     probas, _ = run_ovr_inference(dataset.X, artifacts)
#     if probas is None or not isinstance(probas, dict):
#         raise ValueError("Stage-1 inference did not return dual-head probabilities")
#     return (
#         np.asarray(probas["primary_label"], dtype=float),
#         np.asarray(probas["class_name"], dtype=float),
#         np.asarray(dataset.y),
#         np.asarray(dataset.filenames),
#     ), dataset.bags_meta


# def _build_clean_probability_bags(
#     run_name: str,
#     artifacts: DualOneVsRestArtifacts,
#     *,
#     reduced: bool,
# ) -> list[ProbabilityBag]:
#     (primary_proba, class_proba, y, filenames), bags_meta = _predict_head_probabilities(
#         run_name,
#         artifacts,
#         soundscape=False,
#         reduced=reduced,
#     )
#     bag_ids = _bag_ids_for_train_dataset(run_name, bags_meta)
#     bag_primary_labels = collapse_bag_labels(bag_ids, y[:, 1])
#     bag_targets = _single_label_to_multihot(
#         bag_primary_labels,
#         n_classes=len(artifacts.primary_label.label_encoder.classes_),
#     )
#     bag_names = collapse_bag_labels(bag_ids, filenames)
#     return build_probability_bags(
#         primary_proba,
#         class_proba,
#         bag_ids,
#         bag_targets,
#         source_names=bag_names,
#     )


# def _build_soundscape_probability_bags(
#     run_name: str,
#     artifacts: DualOneVsRestArtifacts,
#     *,
#     reduced: bool,
# ) -> list[ProbabilityBag]:
#     (primary_proba, class_proba, y, filenames), bags_meta = _predict_head_probabilities(
#         run_name,
#         artifacts,
#         soundscape=True,
#         reduced=reduced,
#     )
#     bag_ids = _bag_ids_for_soundscape_dataset(bags_meta)
#     bag_targets = collapse_bag_multilabel_targets(
#         bag_ids,
#         y[:, 1, :len(artifacts.primary_label.label_encoder.classes_)],
#     )
#     bag_names = collapse_bag_labels(bag_ids, filenames)
#     return build_probability_bags(
#         primary_proba,
#         class_proba,
#         bag_ids,
#         bag_targets,
#         source_names=bag_names,
#     )


# def _bag_priors(bags: list[ProbabilityBag]) -> tuple[np.ndarray, np.ndarray]:
#     if not bags:
#         raise ValueError("At least one bag is required to estimate priors")
#     primary_prior = np.mean(
#         np.vstack([bag.primary_label_proba for bag in bags]),
#         axis=0,
#     )
#     class_prior = np.mean(
#         np.vstack([bag.class_name_proba for bag in bags]),
#         axis=0,
#     )
#     return np.asarray(primary_prior, dtype=float), np.asarray(class_prior, dtype=float)


# def _mix_selected_bags(
#     selected_bags: list[ProbabilityBag],
#     *,
#     rng: np.random.Generator,
#     augmentation_cfg,
#     primary_fill_value: np.ndarray,
#     class_fill_value: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     target_length = int(
#         rng.choice(
#             np.asarray([bag.primary_label_proba.shape[0] for bag in selected_bags], dtype=int)
#         )
#     )

#     primary_stack = np.stack(
#         [
#             align_probability_sequence(
#                 bag.primary_label_proba,
#                 target_length,
#                 fill_value=primary_fill_value,
#                 rng=rng,
#             )
#             for bag in selected_bags
#         ],
#         axis=0,
#     )
#     class_stack = np.stack(
#         [
#             align_probability_sequence(
#                 bag.class_name_proba,
#                 target_length,
#                 fill_value=class_fill_value,
#                 rng=rng,
#             )
#             for bag in selected_bags
#         ],
#         axis=0,
#     )

#     if augmentation_cfg.mixture_method == "max":
#         mixed_primary = np.max(primary_stack, axis=0)
#         mixed_class = np.max(class_stack, axis=0)
#     else:
#         weights = rng.uniform(
#             augmentation_cfg.weight_min,
#             augmentation_cfg.weight_max,
#             size=len(selected_bags),
#         ).astype(float)
#         weights /= weights.sum()
#         mixed_primary = np.tensordot(weights, primary_stack, axes=(0, 0))
#         mixed_class = np.tensordot(weights, class_stack, axes=(0, 0))

#     if rng.random() <= augmentation_cfg.temporal_mask_prob:
#         mask = sample_temporal_mask(
#             target_length,
#             rng=rng,
#             keep_ratio_min=augmentation_cfg.temporal_keep_ratio_min,
#             keep_ratio_max=augmentation_cfg.temporal_keep_ratio_max,
#             min_segments=augmentation_cfg.temporal_mask_min_segments,
#             max_segments=augmentation_cfg.temporal_mask_max_segments,
#         )
#         mixed_primary = np.asarray(mixed_primary, dtype=float)
#         mixed_class = np.asarray(mixed_class, dtype=float)
#         mixed_primary[mask] = primary_fill_value
#         mixed_class[mask] = class_fill_value

#     mixed_primary = apply_probability_noise(
#         mixed_primary,
#         sigma=augmentation_cfg.gaussian_noise_std,
#         rng=rng,
#     )
#     mixed_class = apply_probability_noise(
#         mixed_class,
#         sigma=augmentation_cfg.gaussian_noise_std,
#         rng=rng,
#     )

#     target = np.max(
#         np.vstack([bag.primary_target for bag in selected_bags]),
#         axis=0,
#     ).astype(np.int8)
#     return mixed_primary, mixed_class, target


# def _build_augmented_probability_bags(
#     clean_bags: list[ProbabilityBag],
#     soundscape_bags: list[ProbabilityBag],
#     *,
#     augmentation_cfg,
# ) -> list[ProbabilityBag]:
#     if not clean_bags:
#         return []

#     rng = np.random.default_rng(augmentation_cfg.random_state)
#     primary_prior, class_prior = _bag_priors(clean_bags)
#     if augmentation_cfg.temporal_fill_value == "zero":
#         primary_fill_value = np.zeros_like(primary_prior)
#         class_fill_value = np.zeros_like(class_prior)
#     else:
#         primary_fill_value = primary_prior
#         class_fill_value = class_prior

#     mixture_size_probabilities = dict(augmentation_cfg.mixture_size_probabilities)
#     if not mixture_size_probabilities and soundscape_bags:
#         mixture_size_probabilities = estimate_mixture_size_distribution(
#             np.vstack([bag.primary_target for bag in soundscape_bags]),
#             max_mixture_size=augmentation_cfg.max_mixture_size,
#         )
#     if not mixture_size_probabilities:
#         mixture_size_probabilities = DEFAULT_MIXTURE_SIZE_PROBABILITIES
#     support, probs = normalize_mixture_size_distribution(
#         mixture_size_probabilities,
#         max_mixture_size=augmentation_cfg.max_mixture_size,
#     )

#     n_augmented = int(np.ceil(len(clean_bags) * float(augmentation_cfg.synthetic_bag_multiplier)))
#     augmented_bags: list[ProbabilityBag] = []
#     for bag_id in range(n_augmented):
#         mix_size = int(rng.choice(support, p=probs))
#         selected_idx = rng.choice(
#             len(clean_bags),
#             size=mix_size,
#             replace=len(clean_bags) < mix_size,
#         )
#         selected_bags = [clean_bags[int(idx)] for idx in np.asarray(selected_idx, dtype=int)]
#         mixed_primary, mixed_class, target = _mix_selected_bags(
#             selected_bags,
#             rng=rng,
#             augmentation_cfg=augmentation_cfg,
#             primary_fill_value=primary_fill_value,
#             class_fill_value=class_fill_value,
#         )
#         augmented_bags.append(
#             ProbabilityBag(
#                 bag_id=bag_id,
#                 primary_label_proba=mixed_primary,
#                 class_name_proba=mixed_class,
#                 primary_target=target,
#                 source_name=f"synthetic_mix_{mix_size}",
#             )
#         )
#     return augmented_bags


# def _bags_to_training_matrices(
#     bags: list[ProbabilityBag],
#     *,
#     feature_names: list[str],
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     n_features = len(feature_names)
#     if not bags:
#         return (
#             np.empty((0, n_features), dtype=float),
#             np.empty((0, 0), dtype=np.int8),
#             np.empty((0,), dtype=object),
#         )

#     X = np.empty((len(bags), n_features), dtype=np.float32)
#     y = np.empty((len(bags), bags[0].primary_target.shape[0]), dtype=np.int8)
#     source_names = np.empty(len(bags), dtype=object)

#     for row_idx, bag in enumerate(bags):
#         row_features = build_dual_head_bag_feature_matrix(
#             bag.primary_label_proba,
#             bag.class_name_proba,
#             np.zeros(bag.primary_label_proba.shape[0], dtype=int),
#         )
#         X[row_idx] = row_features[0]
#         y[row_idx] = bag.primary_target
#         source_names[row_idx] = bag.source_name

#     return X, y, source_names


# def train_and_save_second_stage_mil_ovr(
#     run_name: str,
#     experiment_name: str,
#     *,
#     model_filename: str | None = None,
#     reduced: bool = True,
# ) -> Path:
#     paths = load_project_paths()
#     experiment_cfg, config_path = load_experiment_config(run_name, experiment_name)
#     if not experiment_cfg.second_stage.enabled:
#         raise ValueError(
#             "second_stage.enabled=false in experiment config. "
#             "Enable second_stage before training."
#         )

#     experiment_dir = paths.experiment_dir(run_name, experiment_name)
#     experiment_dir.mkdir(parents=True, exist_ok=True)
#     second_stage_dir = experiment_dir / "second_stage"
#     second_stage_dir.mkdir(parents=True, exist_ok=True)

#     stage1_model_path = _resolve_source_model_path(
#         experiment_dir,
#         artifact_stem(experiment_cfg),
#         model_filename,
#     )
#     stage1_artifacts = _load_stage_one_artifacts(stage1_model_path)

#     clean_bags = _build_clean_probability_bags(
#         run_name,
#         stage1_artifacts,
#         reduced=reduced,
#     )
#     soundscape_bags = _build_soundscape_probability_bags(
#         run_name,
#         stage1_artifacts,
#         reduced=reduced,
#     )
#     augmented_bags = _build_augmented_probability_bags(
#         clean_bags,
#         soundscape_bags,
#         augmentation_cfg=experiment_cfg.second_stage.augmentation,
#     )

#     training_bags: list[ProbabilityBag] = []
#     if experiment_cfg.second_stage.augmentation.include_clean_bags:
#         training_bags.extend(clean_bags)
#     if experiment_cfg.second_stage.augmentation.include_soundscape_bags:
#         training_bags.extend(soundscape_bags)
#     training_bags.extend(augmented_bags)
#     if not training_bags:
#         raise ValueError("Second-stage training set is empty after applying config")

#     feature_names = build_dual_head_bag_feature_names(
#         stage1_artifacts.primary_label.label_encoder.classes_,
#         stage1_artifacts.class_name.label_encoder.classes_,
#     )
#     X_train, y_train, source_names = _bags_to_training_matrices(
#         training_bags,
#         feature_names=feature_names,
#     )

#     active_label_indices = np.flatnonzero(np.sum(y_train, axis=0) > 0)
#     if active_label_indices.size == 0:
#         raise ValueError("No positive primary labels found for second-stage training")
#     y_train_active = y_train[:, active_label_indices]

#     estimator = build_second_stage_estimator(experiment_cfg.second_stage)
#     model = OneVsRestClassifier(
#         estimator,
#         n_jobs=experiment_cfg.second_stage.n_jobs,
#     )
#     print("Dataset shape is", X_train.shape, y_train_active)
#     model.fit(X_train, y_train_active)

#     summary = {
#         "run_name": run_name,
#         "experiment_name": experiment_name,
#         "config_source": str(config_path.resolve()),
#         "source_model_path": str(stage1_model_path.resolve()),
#         "n_training_bags": int(X_train.shape[0]),
#         "n_features": int(X_train.shape[1]),
#         "n_active_labels": int(active_label_indices.shape[0]),
#         "n_clean_bags": int(len(clean_bags)),
#         "n_soundscape_bags": int(len(soundscape_bags)),
#         "n_augmented_bags": int(len(augmented_bags)),
#         "second_stage": asdict(experiment_cfg.second_stage),
#     }

#     artifacts = SecondStagePrimaryLabelArtifacts(
#         model=model,
#         label_encoder=stage1_artifacts.primary_label.label_encoder,
#         active_label_indices=np.asarray(active_label_indices, dtype=np.int32),
#         feature_names=np.asarray(feature_names, dtype=object),
#         source_model_path=str(stage1_model_path.resolve()),
#         training_summary=summary,
#     )

#     output_stem = "ebm_ovr_mil_second_stage_primary_label"
#     joblib.dump(artifacts, second_stage_dir / f"{output_stem}.joblib")
#     np.save(second_stage_dir / "X_train.npy", X_train)
#     np.save(second_stage_dir / "y_primary_label.npy", y_train)
#     np.save(second_stage_dir / "active_label_indices.npy", active_label_indices)
#     np.save(second_stage_dir / "feature_names.npy", np.asarray(feature_names, dtype=object))
#     np.save(second_stage_dir / "bag_sources.npy", source_names, allow_pickle=True)

#     with open(second_stage_dir / "summary.yaml", "w", encoding="utf-8") as handle:
#         yaml.safe_dump(summary, handle, sort_keys=False)
#     with open(second_stage_dir / "config_source.txt", "w", encoding="utf-8") as handle:
#         handle.write(str(config_path.resolve()))
#     with open(second_stage_dir / "source_model_path.txt", "w", encoding="utf-8") as handle:
#         handle.write(str(stage1_model_path.resolve()))

#     return second_stage_dir
