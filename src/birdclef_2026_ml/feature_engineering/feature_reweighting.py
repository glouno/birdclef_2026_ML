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


def rescale_profile_weights(weights: np.ndarray) -> np.ndarray:
    """Central hook for future weight transforms; raw profile weights for now."""
    return np.asarray(weights, dtype=float)


def load_primary_label_weights(
    train_path=PATHS["proc_train"],
    profiles_path=PATHS["profiles"] / "all_profiles.npy",
) -> dict[str, np.ndarray]:
    train_df = pd.read_parquet(train_path)
    profiles = np.load(profiles_path)
    weights_df = compute_weights(train_df, profiles)
    return {
        str(row.primary_label): rescale_profile_weights(row.profile_norm)
        for row in weights_df.itertuples(index=False)
    }


def infer_mel_band_indices(
    feature_names: list[str] | None,
    n_features: int,
    mel_feature_name: str = MEL_FEATURE_NAME,
) -> np.ndarray:
    if feature_names is None:
        if n_features == 128:
            return np.arange(n_features, dtype=int)
        return np.full(n_features, -1, dtype=int)

    if len(feature_names) != n_features:
        raise ValueError(
            "feature_names length must match n_features "
            f"({len(feature_names)} != {n_features})"
        )

    mel_band_indices = np.full(n_features, -1, dtype=int)
    prefix = f"{mel_feature_name}_"
    for feature_idx, feature_name in enumerate(feature_names):
        name = str(feature_name)
        if not name.startswith(prefix):
            continue
        band_idx = int(name[len(prefix):].split("_", 1)[0]) - 1
        mel_band_indices[feature_idx] = band_idx
    return mel_band_indices


def build_feature_scale_vector(
    band_weights: np.ndarray,
    n_features: int,
    feature_names: list[str] | None = None,
    mel_feature_name: str = MEL_FEATURE_NAME,
) -> np.ndarray:
    weights = np.asarray(band_weights, dtype=float).ravel()
    mel_band_indices = infer_mel_band_indices(
        feature_names=feature_names,
        n_features=n_features,
        mel_feature_name=mel_feature_name,
    )
    scale_vector = np.ones(n_features, dtype=float)
    mel_mask = mel_band_indices >= 0
    scale_vector[mel_mask] = weights[mel_band_indices[mel_mask]]
    return scale_vector
