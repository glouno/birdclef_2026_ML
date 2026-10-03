# BirdCLEF 2026 - Classical ML Pipeline

Repository for BirdCLEF+ 2026 using a classical (non-deep-learning) machine learning pipeline only.

We combine audio features and species-specific frequency profiles with hierarchical
one-vs-rest classifiers, then use soundscape context or multiple-instance learning
(MIL) to handle recordings containing several species.

By [Paul Béglin](https://github.com/glouno) and [BshKatrin](https://github.com/BshKatrin).
See the [project report](rapport.pdf) for experiments, results, and limitations.

## Setup

Requires **Python 3.13+** and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Download the [BirdCLEF+ 2026 competition data](https://www.kaggle.com/competitions/birdclef-2026/data)
separately and place `train.csv`, `train_soundscapes_labels.csv`, `taxonomy.csv`,
`sample_submission.csv`, `train_audio/`, and `train_soundscapes/` under `data/raw/`.
Paths can be changed in `configs/project.yaml`.

Two example runs are included: [overlapping chunks](artifacts/models/runs/experiments/run_chunks_overlap/pipeline.yaml)
and [MIL bags](artifacts/models/runs/experiments/run_mil/pipeline.yaml), each with an
`sgd_baseline.yaml` experiment. Use `run_chunks_overlap` or `run_mil` for `<run-name>`
in the commands below.

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
3. Trim silence on clean train audio.
4. Compute mel-spectrograms without time pooling. This accelerates the feature-building pipeline by allowing you to slice over time and experiment with different chunk durations. All other extracted audio features are computed from these mel-spectrograms.
5. Calculate `primary_label` profiles (the idea is that different species operate at different frequencies).
6. Define the run configuration (`FeatureConfig`, `ChunkConfig`, `MILConfig`) and extract all features, aggregating them per time chunk.
7. Reduce the number of features, e.g., by reducing the dimensionality of the mel-spectrogram from 128 to 32 mels. The training command below expects reduced matrices.
8. Train an `SGDClassifier` model to predict both `class_name` and `primary_label` (hierarchical approach).
9. Run OVR inference (probabilities will be used afterwards).
10. Run exactly one second-stage path (soundscape context or MIL).

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

##### 3. Trim Silence on Clean Train Audio

```bash
uv run python -m birdclef_2026_ml trim-train-silence
```

The input is `data/interim/spectral_gating/train_audio`, and the output is written to
`data/interim/spectral_gating_trim/train_audio`.

##### 4. Compute Mel-Spectrograms Without Time Pooling

Train:

```bash
uv run python -m birdclef_2026_ml extract-mel-spectograms --dataset train --audio-stage clean_trim
```

Soundscapes:

```bash
uv run python -m birdclef_2026_ml extract-mel-spectograms --dataset soundscapes --audio-stage clean
```

This writes raw log-mel `.npy` files to `data/interim/features/mel/...`.

##### 5. Calculate `primary_label` Profiles

Build profiles using global median pooling and, by default, trimmed mean with a threshold of 0.1.

```bash
uv run python -m birdclef_2026_ml build-profiles
```

This command writes:

- `data/processed/features/profiles/species_profiles.npy`
- `data/processed/features/profiles/species_profile_ids.npy`

##### 6. Extract All Features and Aggregate Per Time Chunk

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
  --run-name <run-name> \
  --soundscapes
```

Feature matrices are stored under `artifacts/models/runs/<run-name>/data/full/`.
Soundscape outputs are suffixed with `_soundscape` (for example, `X_soundscape.dat`
and `y_soundscape.dat`). Shapes, dtypes, feature names, and file IDs are recorded
in `metadata.json` or `metadata_soundscape.json`.
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

##### 7. Reduce the Number of Features

The current training CLI loads reduced matrices by default, so run this step
before training. Reduce soundscape matrices too when using the default inference
and second-stage commands.

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
  --target-mel-bins 32 \
  --soundscapes
```

This writes reduced matrices and metadata under
`artifacts/models/runs/<run-name>/data/reduced/`, using the same filenames as the
full matrices.

##### 8. Train an `SGDClassifier` Model to Predict Both `class_name` and `primary_label` (Hierarchical Approach)

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

##### 9. Run OVR inference

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
- `metadata.json`, containing the probability arrays' shapes and dtypes
- `<model-stem>_class_name_preds.npy`, `<model-stem>_primary_label_preds.npy`

When `--soundscapes` is used, outputs are suffixed with `_soundscape` before the
`_class_name_*` or `_primary_label_*` suffixes.

##### 10A. Soundscape second-stage (context features, no MIL)

Trains a soundscape OOF model using stage-1 class/primary probabilities, their
soft-combined probabilities, and context features (site/time).

```bash
uv run python -m birdclef_2026_ml run-soundscape-second-stage \
  --run-name <run-name> \
  --experiment sgd_baseline
```

Optional flags:

- `--model-filename <file.joblib>` to use a non-default saved OVR artifact
- `--n-splits <int>` to control grouped OOF fold count
- `--batch-size <int>` to control stage-1 inference batch size
- `--full` to use full (non-reduced) soundscape feature matrices

Outputs are saved in:

- `artifacts/models/runs/<run-name>/experiments/<experiment-name>/soundscapes/`
- `oof_proba.npy`
- `combined_proba.npy`

##### 10B. MIL second stage on clean-audio bags

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
