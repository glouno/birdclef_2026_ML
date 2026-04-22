# BirdCLEF 2026 - ML Challenge

We can only use Machine Learning models, no deep learning.

## Data layout

- Original files are stored in `data/raw`.
- Processed artifacts are stored in `data/processed`.

## Preprocess train.csv

To clean `train.csv`, run:

```bash
uv run python -m birdclef_2026_ml preprocess-train-for-models
```

To clean `soundscpaes.csv`, run:

```bash
uv run python -m birdclef_2026_ml preprocess-soundscapes-for-models

```

#### After preprocessing

To apply spectral gating (SNR, soft gating version) on **train**, run:

```bash
uv run python -m birdclef_2026_ml spectral-gating --df-input-path=proc_train --input-root=train_audio_dir --output-root=train_audio_spectral_gating_dir
```

To apply spectral gating (SNR, soft gating version) on **soundscapes**, run:

```bash
uv run python -m birdclef_2026_ml spectral-gating --df-input-path=proc_soundscapes --input-root=train_soundscapes_dir --output-root=train_soundscapes_spectral_gating_dir
```
