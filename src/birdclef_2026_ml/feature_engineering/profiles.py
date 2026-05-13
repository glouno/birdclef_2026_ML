from pathlib import Path
import numpy as np


def median_pooling(S: np.ndarray) -> np.ndarray:
    return np.median(S, axis=1)


def trimmed_mean_pooling(S: np.ndarray, trim_ratio: float = 0.1) -> np.ndarray:
    if not 0 <= trim_ratio < 0.5:
        raise ValueError(f"trim_ratio must be in [0, 0.5), got {trim_ratio}")

    n_frames = S.shape[1]
    trim_count = int(n_frames * trim_ratio)
    if trim_count == 0:
        return np.mean(S, axis=1)

    S_sorted = np.sort(S, axis=1)
    return np.mean(S_sorted[:, trim_count:n_frames - trim_count], axis=1)


def mean_pooling(S: np.ndarray) -> np.ndarray:
    return np.mean(S, axis=1)


def species_pooling(S: np.ndarray, method: str = "median", trim_ratio: float = 0.1) -> np.ndarray:
    if method == "median":
        return median_pooling(S)
    if method == "trimmed_mean":
        return trimmed_mean_pooling(S, trim_ratio=trim_ratio)
    if method == "mean":
        return mean_pooling(S)
    raise ValueError(f"Unknown method: {method}")


def get_feature_indices(
    feature_names: list[str],
    prefix: str,
    suffix: str,
) -> np.ndarray:
    indices = np.full(len(feature_names), -1, dtype=int)

    for feature_idx, feature_name in enumerate(feature_names):
        name = str(feature_name)
        if name.startswith(prefix) and name.endswith(suffix):
            band_token = name[len(prefix):].split("_", 1)[0]
            if band_token.isdigit():
                indices[feature_idx] = int(band_token) - 1
    return indices >= 0


def build_profiles(
    df,
    input_mel_dir: Path,
    output_path: Path,
    filename_col: str = "filename",
    pooling_method: str = "trimmed_mean",
    **kwargs
):
    output_path.mkdir(parents=True, exist_ok=True)
    # filenames = df.head(10)[filename_col].unique()
    species_ids = []
    profiles = []

    for group, filenames in df.groupby("primary_label_int")["filename"]:
        S_pooled = []
        print(f"Group {int(group)}")
        for i, filename in enumerate(filenames, start=1):
            mel_path = (input_mel_dir / filename).with_suffix(".npy")
            S = np.load(mel_path)
            S_pooled.append(median_pooling(S))

        species_ids.append(group)
        S_pooled = np.vstack(S_pooled).T

        profile = species_pooling(S_pooled, method=pooling_method, **kwargs)
        profiles.append(profile)

    profiles = np.vstack(profiles, dtype=np.float32)
    species_ids = np.array(species_ids, dtype=np.int32)

    print(profiles)
    print(species_ids)

    return species_ids, profiles


def compute_profile_cosine_similarity(
        X: np.ndarray,
        feature_names: list[str] | np.ndarray,
        species_profiles: np.ndarray,
        species_ids: np.ndarray,
        prefix: str = "mel_spectrogram_",
        suffix: str = "_mean",
) -> tuple[np.ndarray, list[str]]:
    mel_mask = get_feature_indices(list(feature_names), prefix=prefix, suffix=suffix)
    if not mel_mask.any():
        raise ValueError("No mel_spectrogram mean features found in feature_names.")

    X_mel = np.asarray(X[:, mel_mask], dtype=float)
    profiles = np.asarray(species_profiles, dtype=float)
    X_mel = X_mel / np.clip(np.linalg.norm(X_mel, axis=1, keepdims=True), 1e-12, None)
    profiles = profiles / np.clip(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-12, None)
    similarities = X_mel @ profiles.T

    feature_names_out = [f"cos_mel_spectrogram_mean_{species_id}" for species_id in species_ids]
    return similarities.astype(np.float32), feature_names_out
