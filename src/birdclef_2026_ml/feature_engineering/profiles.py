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


def build_profile(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | np.ndarray,
    pooling_method: str = "median",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    mel_mask = get_feature_indices(
        feature_names=list(feature_names),
        prefix="mel_spectrogram_",
        suffix="_mean",
    )
    if not mel_mask.any():
        raise ValueError("No mel_spectrogram mean features found in feature_names.")

    X_arr = X[:, mel_mask]
    species_ids = y[:, 1]

    unique_species, first_idx = np.unique(species_ids, return_index=True)
    order = np.argsort(first_idx)
    unique_species = unique_species[order]

    profiles = [
        species_pooling(X_arr[species_ids == species_id].T, method=pooling_method, **kwargs)
        for species_id in unique_species
    ]
    return unique_species, np.asarray(profiles)


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


def save_profiles(
    output_dir: Path,
    species_ids: np.ndarray,
    profiles: np.ndarray,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "species_profiles.npy"
    species_ids_path = output_dir / "species_profile_ids.npy"
    np.save(profiles_path, profiles)
    np.save(species_ids_path, species_ids)
    return profiles_path, species_ids_path
