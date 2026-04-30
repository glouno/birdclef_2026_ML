## Overview

Classical ML pipeline for BirdCLEF+ 2026.

Goal: classify bird species from audio with non-deep-learning methods only.

Main data:

- train audio: single-species recordings
- soundscapes: multi-species audio with 5s annotations

## Core Rules

- Use classical ML only. No deep learning models.
- Run Python with `uv run python ...`.
- Do not hardcode filesystem paths. Use YAML-backed path config.
- Prefer updating existing config-driven flows over adding one-off scripts.

## Path And Config System

- Project paths live in `configs/project.yaml`.
- Path resolution code lives in `src/birdclef_2026_ml/paths.py`.
- Load paths via `load_project_paths()`.
- Optional env overrides:
  - `PROJECT_ROOT`
  - `BIRDCLEF_CONFIG`

Do not reintroduce global `PATHS` dict pattern.

## Directory Layout

- `data/raw`: source metadata and raw audio
- `data/interim`: reversible intermediate artifacts
  - spectral gating outputs
  - raw mel caches
- `data/processed`: stable model-ready artifacts
  - processed metadata
  - pooled features
  - label encoders
  - species profiles
- `artifacts/models/runs`: run artifacts
  - memmaps
  - trained models
  - predictions
- `artifacts/models/runs/experiments`: experiment YAML configs by run

## Training Experiments

Experiment config path:

- `artifacts/models/runs/experiments/<run-name>/<experiment-name>.yaml`

Training code:

- `src/birdclef_2026_ml/training/model_training.py`

Train command:

- `uv run python -m birdclef_2026_ml train-ovr-models-chunks --run-name <run> --experiment <name>`

Experiment outputs go to:

- `artifacts/models/runs/<run-name>/experiments/<experiment-name>/`

Keep experiment configs small, explicit, immutable after run when possible.

## Common CLI Flow

1. `preprocess-datasets-for-models`
2. `spectral-gating --dataset <train|soundscapes>`
3. `extract-all-features --dataset <train|soundscapes>`
4. `pool-mel-features --dataset <train|soundscapes>`
5. `build-feature-matrices --dataset ... --run-name ...`
6. `train-ovr-models-chunks --run-name ... --experiment ...`

## Code Areas

- `src/birdclef_2026_ml/cli.py`: CLI entrypoints
- `src/birdclef_2026_ml/configs.py`: typed runtime configs
- `src/birdclef_2026_ml/paths.py`: project path resolver
- `src/birdclef_2026_ml/feature_engineering/`: feature extraction and pooling
- `src/birdclef_2026_ml/processing/`: preprocessing and dataset builders
- `src/birdclef_2026_ml/training/`: training and evaluation

## When Editing

- Prefer semantic args like `dataset`, `audio_stage`, `feature_kind`, `run_name`, `experiment`.
- Keep transient artifacts in `data/interim`, not `data/processed`.
- Keep reusable derived artifacts in `data/processed`.
- Keep run-specific outputs in `artifacts/models/runs`.
- Update `README.md` when CLI, config schema, or artifact layout changes.
