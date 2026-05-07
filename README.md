# BirdCLEF 2026 - Classical ML Pipeline

Repository for BirdCLEF+ 2026 using a classical (non-deep-learning) machine learning pipeline only.

## Project Layout

- `data/raw`: Original competition metadata and audio files
- `data/interim`: Transient audio and cache artifacts
- `data/processed`: Model-ready metadata and stable derived assets
- `artifacts/models/runs`: Feature matrices, trained models, and predictions
- `artifacts/models/runs/experiments`: Experiment YAMLs and trained experiment outputs
- `configs/project.yaml`: Central path configuration

## Path and Config Management

All project paths are resolved from [`configs/project.yaml`](configs/project.yaml).

To override the config file:

```bash
BIRDCLEF_CONFIG=/abs/path/to/project.yaml uv run python -m birdclef_2026_ml ...
```

To override the project root:

```bash
PROJECT_ROOT=/abs/path/to/repo uv run python -m birdclef_2026_ml ...
```

## Preprocess metadata

### Workflow

1. Preprocess `train` and `soundscapes` metadata (convert `.csv` to `.parquet`).
2. Apply spectral gating to both `train` and `soundscapes` audio files.
3. Compute mel-spectrograms without time pooling. This accelerates the feature-building pipeline by allowing you to slice over time and experiment with different chunk durations. All other extracted audio features are computed from these mel-spectrograms.
4. Calculate `primary_label` profiles (the idea is that different species operate at different frequencies).
5. Define the run configuration (`FeatureConfig`, `ChunkConfig`, `MILConfig`) and extract all features, aggregating them per time chunk.
6. [Optional] Reduce the number of features, e.g., by reducing the dimensionality of the mel-spectrogram from 128 to 32 mels.
7. Train an `SGDClassifier` model to predict both `class_name` and `primary_label` (hierarchical approach).
8. Run OVR inference (probabilities will be used afterwards)
<!-- 8. Train a second-stage OVR model on clean-audio MIL bag features to refine `primary_label`.
9. Run soundscape OOF post-processing to calibrate probabilities, tune thresholds, and save multilabel predictions.
10. Run OVR inference to generate predictions. -->

### Commands

##### 1. Preprocess `train` and `soundscapes` metadata (CSV to Parquet)

```bash
uv run python -m birdclef_2026_ml preprocess-datasets-for-models
```

This command writes:

- `data/processed/metadata/train.parquet`
- `data/processed/metadata/train_soundscapes_labels.parquet`
- `data/processed/label_encoders/*.joblib`

##### 2. Apply Spectral Gating to Both `train` and `soundscapes` Audio

Train:

```bash
uv run python -m birdclef_2026_ml spectral-gating --dataset train
```

Soundscapes:

```bash
uv run python -m birdclef_2026_ml spectral-gating --dataset soundscapes
```

The input defaults to raw audio. The output is written to `data/interim/spectral_gating/...`.

##### 3. Compute Mel-Spectrograms Without Time Pooling

Train:

```bash
uv run python -m birdclef_2026_ml extract-mel-spectograms --dataset train
```

Soundscapes:

```bash
uv run python -m birdclef_2026_ml extract-mel-spectograms --dataset soundscapes
```

This writes raw log-mel `.npy` files to `data/interim/features/mel/...`.

##### 4. Calculate `primary_label` Profiles

Build profiles using global median pooling and, by default, trimmed mean with a threshold of 0.1.

```bash
uv run python -m birdclef_2026_ml build-profiles
```

This command writes:

- `data/processed/features/profiles/species_profiles.npy`
- `data/processed/features/profiles/species_profile_ids.npy`

##### 5. Extract All Features and Aggregate Per Time Chunk

Define the pipeline configuration in:

- `artifacts/models/runs/experiments/<run-name>/pipeline.yaml`

To extract features from mel caches:

```bash
uv run python -m birdclef_2026_ml build-feature-matrices \
  --dataset train \
  --feature-kind mel \
  --run-name <run-name>
```

Soundscapes:

```bash
uv run python -m birdclef_2026_ml build-feature-matrices \
  --dataset soundscapes \
  --feature-kind mel \
  --run-name \
  --soundscapes
```

Run artifacts are stored under `artifacts/models/runs/<run-name>`.
Soundscape outputs are suffixed with `_soundscape` (for example, `X_soundscape.dat`,
`y_soundscape.dat`, and `feature_names_soundscape.npy`).
MIL runs also write `bags_meta.npy`, where each row stores `[bag_id, chunk_id_within_bag]`.
Soundscape MIL runs write `bags_meta_soundscape.npy`.

The derived feature families include:

- `mel_spectrogram`, `mel_spectrogram_delta`, `mel_spectrogram_delta2`
- `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_contrast`
- `spectral_entropy`, `spectral_flux`
- `spectral_energy_proxy`, `burstiness`
- `inter_event_prev_gap`, `inter_event_next_gap`
- `glcm_*`
- `lbp_hist`

##### 6. [Optional] Reduce the Number of Features

Train:

```bash
uv run python -m birdclef_2026_ml reduce-feature-matrices \
  --run-name <run-name> \
  --target-mel-bins 32
```

Soundscapes:

```bash
uv run python -m birdclef_2026_ml reduce-feature-matrices \
  --run-name <run-name> \
  --target-mel-bins 32
  --soundscapes
```

This writes reduced copies suffixed with `_reduced` in the same run directory:

##### 7. Train an `SGDClassifier` Model to Predict Both `class_name` and `primary_label` (Hierarchical Approach)

###### Train One-vs-Rest (OVR) Models

Define the experiment configuration in:

- `artifacts/models/runs/experiments/<run-name>/<experiment-name>.yaml`

```bash
uv run python -m birdclef_2026_ml train-ovr-models-chunks \
  --run-name <run-name> \
  --experiment sgd_baseline
```

Outputs are saved in:

- `artifacts/models/runs/<run-name>/experiments/<experiment-name>/`
- Includes the trained model, optional `val_indices.npy`, and a resolved config copy

The training configuration supports custom batch early stopping in the OVR loop:

- `training.early_stopping`: enable validation-based stopping
- `training.n_iter_no_change`: stop after this many epochs without val log-loss improvement
- `training.tol`: minimum val log-loss improvement to reset patience

When `training.early_stopping: true`, ensure `training.train_val_split: true` or provide validation data through the calling code.

##### 8. Train MIL Second Stage on Clean-Audio Bags

Train command:

```bash
uv run python -m birdclef_2026_ml train-mil-second-stage-clean \
  --run-name <run-name> \
  --experiment sgd_baseline
```

Or run the two steps separately:

```bash
uv run python -m birdclef_2026_ml build-mil-second-stage-clean-data \
  --run-name <run-name> \
  --experiment sgd_baseline
```

Add `--soundscapes` to use soundscape MIL matrices (bag ids come from `bags_meta_soundscape.npy`).

```bash
uv run python -m birdclef_2026_ml train-mil-second-stage-clean-model \
  --run-name <run-name> \
  --experiment sgd_baseline
```

To run the trained clean-audio second-stage model on soundscape bag features:

```bash
uv run python -m birdclef_2026_ml run-mil-second-stage-clean-soundscape \
  --run-name <run-name> \
  --experiment sgd_baseline
```

To warm-start the clean-audio second-stage models and score soundscape OOF splits:

```bash
uv run python -m birdclef_2026_ml run-mil-second-stage-clean-soundscape-oof \
  --run-name <run-name> \
  --experiment sgd_baseline
```

To train a soundscape OOF model using MIL soundscape probabilities + context features:

```bash
uv run python -m birdclef_2026_ml run-soundscape-mil-context-oof \
  --run-name <run-name> \
  --experiment sgd_baseline
```

This uses P(bag), per-class bag aggregation features, and context features. Classes with no
positives in a fold default back to P(bag).

This stage:

- loads stage-1 `class_name` and `primary_label` probabilities from the experiment inference folder
- builds per-primary-label bag features using summary statistics of `P(primary_label)` and its parent `P(class_name)`
- trains a one-vs-rest second-stage model on train bags and evaluates on `val_indices.npy`

Outputs are saved in:

- `artifacts/models/runs/<run-name>/experiments/<experiment-name>/clean_audio/`
- `train_bag_features_primary.npy`, `val_bag_features_primary.npy` (shape: `n_primary x n_bags x n_features`)
- `train_bag_features_primary_soundscape.npy` (when `--soundscapes` is used)
- `train_bag_labels.npy`, `val_bag_labels.npy`
- `train_bag_labels_soundscape.npy`, `val_bag_labels_soundscape.npy` (when `--soundscapes` is used)
- `train_bag_ids.npy`, `val_bag_ids.npy`
- `train_bag_ids_soundscape.npy`, `val_bag_ids_soundscape.npy` (when `--soundscapes` is used)
- `val_proba.npy`
- `<model-stem>_mil_second_stage_primary.joblib`
- `<model-stem>_mil_second_stage_primary_soundscape_proba.npy`
- `<model-stem>_mil_second_stage_primary_soundscape_oof.npy`
- `<model-stem>_mil_second_stage_primary_soundscape_context_oof.npy`

---

## Run soundscape OOF post-processing

This step loads trained `DualOneVsRestArtifacts`, predicts `class_name` and
`primary_label` probabilities in log space on soundscapes, combines both heads,
fits per-class probability calibrators with grouped out-of-fold splits,
tunes per-class multilabel thresholds for macro F1, then saves OOF and final
predictions under the experiment soundscape directory.

```bash
uv run python -m birdclef_2026_ml run-soundscape-oof \
  --run-name <run-name> \
  --experiment sgd_baseline \
  --calibration calibration_sigmoid
```

Optional flags:

- `--model-filename <file.joblib>` to use non-default saved OVR artifact
- `--n-splits <int>` to control grouped OOF fold count
- `--random-state <int>` to control OOF split seed
- `--max-rounds <int>` to control threshold tuning passes
- `--full` to use full (non-reduced) soundscape matrices

Outputs are saved in:

- `artifacts/models/runs/<run-name>/experiments/<experiment-name>/soundscapes/`
- `oof/class_name_combined_log_proba.npy`
- `oof/class_name_calibrated_proba.npy`
- `oof/class_name_thresholds.npy`
- `oof/class_name_predictions.npy`
- `oof/primary_label_combined_log_proba.npy`
- `oof/primary_label_calibrated_proba.npy`
- `oof/primary_label_thresholds.npy`
- `oof/primary_label_predictions.npy`
- `final/class_name_combined_log_proba.npy`
- `final/class_name_calibrated_proba.npy`
- `final/class_name_thresholds.npy`
- `final/class_name_predictions.npy`
- `final/primary_label_combined_log_proba.npy`
- `final/primary_label_calibrated_proba.npy`
- `final/primary_label_thresholds.npy`
- `final/primary_label_predictions.npy`
- `fold_<k>_train_indices.npy`, `fold_<k>_val_indices.npy`
- `metrics.yaml`
- `<model-stem>_soundscape_oof.joblib`

## Run OVR inference

Inference loads a saved `.joblib` artifact and writes predictions under the
experiment's `inference` subfolder.

```bash
uv run python -m birdclef_2026_ml run-ovr-inference \
  --run-name <run-name> \
  --experiment sgd_baseline \
  --soundscapes
```

Optional flags:

- `--model-filename <file.joblib>` to infer from a specific artifact
- `--full` to use full (non-reduced) feature matrices

Outputs are saved in:

- `artifacts/models/runs/<run-name>/experiments/<experiment-name>/inference/`
- `<model-stem>_class_name_probas.dat`, `<model-stem>_primary_label_probas.dat`
- `<model-stem>_class_name_probas_shape.npy`, `<model-stem>_class_name_probas_dtype.npy`
- `<model-stem>_primary_label_probas_shape.npy`, `<model-stem>_primary_label_probas_dtype.npy`
- `<model-stem>_class_name_preds.npy`, `<model-stem>_primary_label_preds.npy`

When `--soundscapes` is used, outputs are suffixed with `_soundscape` before the
`_class_name_*` or `_primary_label_*` suffixes.

## Run MIL inference

MIL inference aggregates window-level OVR probabilities into bag-level
probabilities and predictions.

```bash
uv run python -m birdclef_2026_ml run-mil-inference \
  --run-name <run-name> \
  --experiment sgd_baseline \
  --aggregation mean
```

Optional flags:

- `--model-filename <file.joblib>` to use non-default saved OVR artifact
- `--soundscapes` to aggregate soundscape windows
- `--full` to use full (non-reduced) feature matrices
- `--val-idx-path <path.npy>` to remap validation indices to bag ids

Outputs are saved in:

- `artifacts/models/runs/<run-name>/experiments/<experiment-name>/inference/`
- `<model-stem>_mil_<aggregation>_bag_ids.npy`
- `<model-stem>_mil_<aggregation>_val_bag_ids.npy` (if `--val-idx-path` is provided)
- `<model-stem>_mil_<aggregation>_val_bag_idx.npy` (if `--val-idx-path` is provided)
- `<model-stem>_mil_<aggregation>_class_name_probas.npy`
- `<model-stem>_mil_<aggregation>_primary_label_probas.npy`
- `<model-stem>_mil_<aggregation>_class_name_preds.npy`
- `<model-stem>_mil_<aggregation>_primary_label_preds.npy`
- `<model-stem>_mil_<aggregation>_class_name_true.npy`
- `<model-stem>_mil_<aggregation>_primary_label_true.npy`

---

## Calibrate OVR models

Calibration is a separate post-training step. It loads the pretrained OVR artifact,
uses the saved validation fold from the experiment, and writes a calibrated copy
without overwriting the original model.

Requirements:

- `training.train_val_split: true` in the experiment config used for training
- calibration YAML stored in `artifacts/models/runs/experiments/<run-name>/`

Example calibration configuration:

`artifacts/models/runs/experiments/<run-name>/calibration_sigmoid.yaml`

```bash
uv run python -m birdclef_2026_ml calibrate-ovr-models-chunks \
  --run-name <run-name> \
  --experiment sgd_baseline \
  --calibration calibration_sigmoid
```

Additional outputs are saved in the same experiment directory:

- `<model-stem>_ovr_calibrated_<calibration-name>.joblib`
- `<model-stem>_val_probas_calibrated_<calibration-name>.npy`
- `<model-stem>_val_preds_calibrated_<calibration-name>.npy`
- `calibration_config_source_<calibration-name>.txt`

## Tune OVR thresholds

Threshold tuning is a separate post-training step. It loads saved `.joblib` OVR
artifacts, searches for per-class probability thresholds on the saved validation fold,
maximizes the chosen score, and then saves new threshold-tuned artifacts.

Requirements:

- `training.train_val_split: true` in experiment config used for training

Default score:

- `macro_f1`

Supported scores:

- `macro_f1`
- `micro_f1`
- `weighted_f1`
- `accuracy`
- `balanced_accuracy`

To tune the default saved dual artifact:

```bash
uv run python -m birdclef_2026_ml tune-ovr-thresholds \
  --run-name <run-name> \
  --experiment sgd_baseline \
  --score macro_f1
```

To tune a calibrated artifact:

```bash
uv run python -m birdclef_2026_ml tune-ovr-thresholds \
  --run-name <run-name> \
  --experiment sgd_baseline \
  --model-filename sgdclassifier_ovr_calibrated_calibration_sigmoid.joblib \
  --score macro_f1
```

If the input `.joblib` contains `OneVsRestArtifacts`, pass the target:

```bash
uv run python -m birdclef_2026_ml tune-ovr-thresholds \
  --run-name <run-name> \
  --experiment sgd_baseline \
  --model-filename some_single_target_model.joblib \
  --target-name primary_label \
  --score macro_f1
```

Outputs are saved in the same experiment directory:

- `<input-model-stem>_threshold_tuned_<score>.joblib`
- `<input-model-stem>_threshold_tuned_<score>_val_preds.npy`
- `<input-model-stem>_threshold_tuned_<score>_val_probas.npy`
- `<input-model-stem>_threshold_tuned_<score>_thresholds.npy`
