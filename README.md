# BirdCLEF 2026 - Classical ML Pipeline

Repo for BirdCLEF+ 2026 with non-deep-learning pipeline only.

## Project layout

- `data/raw`: original competition metadata + audio
- `data/interim`: transient audio/cache artifacts
- `data/processed`: model-ready metadata + stable derived assets
- `artifacts/models/runs`: feature matrices, trained runs, predictions
- `artifacts/models/runs/experiments`: experiment YAMLs + trained experiment outputs
- `configs/project.yaml`: central path config

Use `data/interim` for reversible preprocessing. Spectral gating output lives there now.

## Path/config management

All project paths resolved from [`configs/project.yaml`](configs/project.yaml).

Override config file:

```bash
BIRDCLEF_CONFIG=/abs/path/to/project.yaml uv run python -m birdclef_2026_ml ...
```

Override project root:

```bash
PROJECT_ROOT=/abs/path/to/repo uv run python -m birdclef_2026_ml ...
```

## Preprocess metadata

```bash
uv run python -m birdclef_2026_ml preprocess-datasets-for-models
```

Writes:

- `data/processed/metadata/train.parquet`
- `data/processed/metadata/train_soundscapes_labels.parquet`
- `data/processed/label_encoders/*.joblib`

## Spectral gating

Train:

```bash
uv run python -m birdclef_2026_ml spectral-gating --dataset train
```

Soundscapes:

```bash
uv run python -m birdclef_2026_ml spectral-gating --dataset soundscapes
```

Input defaults to raw audio. Output defaults to `data/interim/spectral_gating/...`.

## Extract mel spectrograms

Train:

```bash
uv run python -m birdclef_2026_ml extract-all-features --dataset train
```

Soundscapes:

```bash
uv run python -m birdclef_2026_ml extract-all-features --dataset soundscapes
```

Writes raw log-mel `.npy` files to `data/interim/features/mel/...`.

## Pool mel features

Train:

```bash
uv run python -m birdclef_2026_ml pool-mel-features --dataset train
```

Soundscapes:

```bash
uv run python -m birdclef_2026_ml pool-mel-features --dataset soundscapes
```

Writes pooled tabular features to `data/processed/features/pooled/...`.

Derived feature families include:

- `mel_spectrogram`, `mel_spectrogram_delta`, `mel_spectrogram_delta2`
- `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_contrast`
- `spectral_entropy`, `spectral_flux`
- `spectral_energy_proxy`, `burstiness`
- `inter_event_prev_gap`, `inter_event_next_gap`
- `glcm_*`
- `lbp_hist`

## Build feature matrices

From mel caches:

```bash
uv run python -m birdclef_2026_ml build-feature-matrices \
  --dataset train \
  --feature-kind mel \
  --run-name run_with_profiles \
  --profiles-path data/processed/profiles/species_profiles.npy \
  --species-ids-path data/processed/profiles/species_profile_ids.npy
```

Soundscapes:

```bash
uv run python -m birdclef_2026_ml build-feature-matrices \
  --dataset soundscapes \
  --feature-kind mel \
  --run-name run_soundscapes
```

Run artifacts stored under `artifacts/models/runs/<run-name>`.

## Build profiles

```bash
uv run python -m birdclef_2026_ml build-profiles --run-name run_with_profiles
```

Writes:

- `data/processed/features/profiles/species_profiles.npy`
- `data/processed/features/profiles/species_profile_ids.npy`

## Reduce feature matrices

```bash
uv run python -m birdclef_2026_ml reduce-feature-matrices \
  --run-name run_with_profiles \
  --target-mel-bins 32
```

Writes reduced copies in same run dir:

- `X_reduced.dat`
- `y_reduced.dat`
- `feature_names_reduced.npy`
- `file_ids_reduced.npy`
- `shape_X_reduced.npy`
- `shape_y_reduced.npy`
- `dtype_X_reduced.npy`
- `dtype_y_reduced.npy`

## Train OVR models

Define experiment config in:

- `artifacts/models/runs/experiments/<run-name>/<experiment-name>.yaml`

Example:

```yaml
model:
  type: SGDClassifier
  params:
    loss: log_loss
    penalty: elasticnet
    alpha: 1.0e-4
    max_iter: 3000
    tol: 1.0e-4
    early_stopping: false
    learning_rate: optimal
    average: true
    random_state: 42

training:
  batch_size: 128
  epochs: 3
  train_val_split: true
```

Best practice:

- keep experiment YAML immutable after run
- one YAML per named experiment
- compare runs by changing few params at time
- store outputs under run-scoped experiment folders

```bash
uv run python -m birdclef_2026_ml train-ovr-models-chunks \
  --run-name run_with_profiles \
  --experiment sgd_baseline
```

Outputs saved in:

- `artifacts/models/runs/<run-name>/experiments/<experiment-name>/`
- includes trained model, val predictions, resolved config copy

## Calibrate OVR models

Calibration is a separate post-training step. It loads the pretrained OVR artifact,
uses the saved validation fold from the experiment, and writes a calibrated copy
without overwriting the original model.

Requirement:

- `training.train_val_split: true` in the experiment config used for training
- calibration YAML stored in `artifacts/models/runs/experiments/<run-name>/`

Example calibration config:

`artifacts/models/runs/experiments/run_with_profiles/calibration_sigmoid.yaml`

```bash
uv run python -m birdclef_2026_ml calibrate-ovr-models-chunks \
  --run-name run_with_profiles \
  --experiment sgd_baseline \
  --calibration calibration_sigmoid
```

Additional outputs saved in the same experiment dir:

- `<model-stem>_ovr_calibrated_<calibration-name>.joblib`
- `<model-stem>_val_probas_calibrated_<calibration-name>.npy`
- `<model-stem>_val_preds_calibrated_<calibration-name>.npy`
- `calibration_config_source_<calibration-name>.txt`

## Tune OVR thresholds

Threshold tuning is separate post-training step. It loads saved `.joblib` OVR
artifacts, searches per-class probability thresholds on saved validation fold,
maximizes chosen score, then saves new threshold-tuned artifacts.

Requirement:

- `training.train_val_split: true` in experiment config used for training

Default score:

- `macro_f1`

Supported scores:

- `macro_f1`
- `micro_f1`
- `weighted_f1`
- `accuracy`
- `balanced_accuracy`

Tune default saved dual artifact:

```bash
uv run python -m birdclef_2026_ml tune-ovr-thresholds \
  --run-name run_with_profiles \
  --experiment sgd_baseline \
  --score macro_f1
```

Tune calibrated artifact:

```bash
uv run python -m birdclef_2026_ml tune-ovr-thresholds \
  --run-name run_with_profiles \
  --experiment sgd_baseline \
  --model-filename sgdclassifier_ovr_calibrated_calibration_sigmoid.joblib \
  --score macro_f1
```

If input `.joblib` contains `OneVsRestArtifacts`, pass target:

```bash
uv run python -m birdclef_2026_ml tune-ovr-thresholds \
  --run-name run_with_profiles \
  --experiment sgd_baseline \
  --model-filename some_single_target_model.joblib \
  --target-name primary_label \
  --score macro_f1
```

Outputs saved in same experiment dir:

- `<input-model-stem>_threshold_tuned_<score>.joblib`
- `<input-model-stem>_threshold_tuned_<score>_val_preds.npy`
- `<input-model-stem>_threshold_tuned_<score>_val_probas.npy`
- `<input-model-stem>_threshold_tuned_<score>_thresholds.npy`
