import pandas as pd
import numpy as np

from birdclef_2026_ml.paths import PATHS


MEL_FEATURE_NAME = "mel_spectrogram"


def compute_weights(df, profiles):
    def compute_median(arrays):
        return np.median(np.stack(arrays), axis=0)

    profiles = np.asarray(profiles, dtype=float)
    profiles_norm = profiles - profiles.max(axis=1, keepdims=True)
    profiles_df = pd.DataFrame(data={
        "primary_label": df["primary_label"],
        "class_name": df["class_name"],
        "common_name": df["common_name"],
        "profile_norm": profiles_norm.tolist(),
    })

    return profiles_df.groupby(
        ["primary_label", "class_name", "common_name"],
        sort=False,
        as_index=False,
    )["profile_norm"].apply(compute_median)


def apply_profile_weights(x: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    # alpha [0.1, 0.5] -> mild effect, stable
    # alpha [0.5, 1] -> noticeable, still stable
    # alpha >1 tends to overfit

    # sum since we work with dB
    print("apply_profile_weights", weights)
    return x + alpha * weights


def normalize_profile_weights_mean_std(weights: np.ndarray) -> np.ndarray:
    """Normalize weights to zero mean and unit standard deviation."""
    weights_arr = np.asarray(weights, dtype=float)
    mean = weights_arr.mean()
    std = weights_arr.std()
    if std == 0.0:
        return weights_arr - mean
    return (weights_arr - mean) / std


def rescale_profile_weights(weights: np.ndarray) -> np.ndarray:
    """Central hook for profile weight transforms."""
    return normalize_profile_weights_mean_std(weights)


def load_weights(
    col: str,
    train_path=PATHS["proc_train"],
    profiles_path=PATHS["profiles"] / "all_profiles.npy",
) -> dict[str, np.ndarray]:
    train_df = pd.read_parquet(train_path)
    profiles = np.load(profiles_path)
    weights_df = compute_weights(train_df, profiles)
    return {
        str(getattr(row, col)): rescale_profile_weights(row.profile_norm)
        for row in weights_df.itertuples(index=False)
    }


def infer_mel_band_indices(
    feature_names: list[str],
) -> np.ndarray:

    mel_band_indices = np.full(len(feature_names), -1, dtype=int)
    prefix = f"{MEL_FEATURE_NAME}_"

    for feature_idx, feature_name in enumerate(feature_names):
        name = str(feature_name)
        if not name.startswith(prefix):
            continue
        band_token = name[len(prefix):].split("_", 1)[0]
        if band_token.isdigit():
            mel_band_indices[feature_idx] = int(band_token) - 1
        break
    return mel_band_indices


def build_feature_scale_vector(
    band_weights: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    weights = np.asarray(band_weights, dtype=float).ravel()
    mel_band_indices = infer_mel_band_indices(
        feature_names=feature_names,
    )
    scale_vector = np.ones(len(feature_names), dtype=float)
    mel_mask = mel_band_indices >= 0
    scale_vector[mel_mask] = weights[mel_band_indices[mel_mask]]
    return scale_vector
