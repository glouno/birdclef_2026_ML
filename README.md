# BirdCLEF 2026 - ML Challenge

We can only use Machine Learning models, no deep learning.

## Data layout

- Original files are stored in `data/raw`.
- Processed artifacts are stored in `data/processed`.

## Preprocess train.csv

To clean `train.csv` and `soundscpaes.csv`, run:

```bash
uv run python -m birdclef_2026_ml preprocess-datasets-for-models
```

#### After preprocessing

### Spectral Gating (SNR, soft gating version)

To apply spectral gating on **train**, run:

```bash
uv run python -m birdclef_2026_ml spectral-gating \
	--df-input-path=proc_train \
	--config-path=data/configs/spectral_gating.json \
	--input-root=train_audio_dir \
	--output-root=train_audio_spectral_gating_dir
```

To apply spectral gating on **soundscapes**, run:

```bash
uv run python -m birdclef_2026_ml spectral-gating \
	--df-input-path=proc_soundscapes \
	--config-path=data/processed/train_soundscapes_spectral_gating/config.json \
	--input-root=train_soundscapes_dir \
	--output-root=train_soundscapes_spectral_gating_dir
```

### Build Audio Profiles

To build all audio profiles and save to all_profiles.npy, run:

```bash
uv run python -m birdclef_2026_ml build-profiles \
	--output-path-key=profiles \
	--n-mels=128 \
	--input-train-key=proc_train \
	--spectral-gating-config-path=data/processed/train_audio_spectral_gating/config.json
```

### Extract all features (before time aggregation)

```bash
uv run python -m birdclef_2026_ml extract-all-features --input-df=proc_train --input-audio-dir=train_audio_spectral_gating_dir --output-audio-dir=train_audio_features_dir

uv run python -m birdclef_2026_ml extract-all-features --input-df=proc_soundscapes --input-audio-dir=train_soundscapes_spectral_gating_dir --output-audio-dir=train_soundscapes_features_dir
```

### Build feature matrices (after feature extraction to allow time slicing)

```bash
uv run python -m birdclef_2026_ml build-feature-matrices --input-audio-dir=train_audio_spectral_gating_dir --input-features-dir=train_audio_features_dir --output-folder=run_1 --input-df=proc_train

uv run python -m birdclef_2026_ml build-feature-matrices --input-audio-dir=train_soundscapes_spectral_gating_dir --input-features-dir=train_soundscapes_features_dir --output-folder=run_1 --input-df=proc_soundscapes
```
