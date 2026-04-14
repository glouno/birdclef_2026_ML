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
