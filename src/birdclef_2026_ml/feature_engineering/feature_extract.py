import numpy as np
import librosa

from birdclef_2026_ml.configs import FeatureConfig
from birdclef_2026_ml.feature_engineering.profiles import compute_profile_cosine_similarity


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(float, copy=True)

    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _moving_std(x: np.ndarray, window: int) -> np.ndarray:
    mean = _moving_average(x, window)
    mean_sq = _moving_average(np.square(x), window)
    return np.sqrt(np.maximum(mean_sq - np.square(mean), 0.0))


def _downsample_freq_axis(S: np.ndarray, target_bins: int) -> np.ndarray:
    if S.shape[0] <= target_bins:
        return S

    edges = np.linspace(0, S.shape[0], target_bins + 1, dtype=int)
    rows = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            end = min(start + 1, S.shape[0])
        rows.append(np.mean(S[start:end], axis=0))
    return np.vstack(rows)


def _quantize_spectrogram(S: np.ndarray, n_levels: int) -> np.ndarray:
    S_min = float(np.min(S))
    S_max = float(np.max(S))
    if S_max <= S_min:
        return np.zeros_like(S, dtype=np.int32)

    scaled = (S - S_min) / (S_max - S_min)
    return np.clip((scaled * (n_levels - 1)).astype(np.int32), 0, n_levels - 1)


def _glcm_from_pairs(src: np.ndarray, dst: np.ndarray, n_levels: int) -> np.ndarray:
    glcm = np.zeros((n_levels, n_levels), dtype=float)
    np.add.at(glcm, (src.ravel(), dst.ravel()), 1.0)
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def _glcm_stats(glcm: np.ndarray) -> tuple[float, float, float, float]:
    n_levels = glcm.shape[0]
    ii, jj = np.indices((n_levels, n_levels))
    contrast = float(np.sum(glcm * np.square(ii - jj)))
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(ii - jj))))
    energy = float(np.sum(np.square(glcm)))
    entropy = float(-np.sum(glcm * _safe_log(glcm)))
    return contrast, entropy, homogeneity, energy


def _lbp_codes(patch: np.ndarray) -> np.ndarray:
    center = patch[1:-1, 1:-1]
    if center.size == 0:
        return np.array([], dtype=np.uint8)

    neighbors = (
        (patch[:-2, :-2] >= center).astype(np.uint8) << 0
        | (patch[:-2, 1:-1] >= center).astype(np.uint8) << 1
        | (patch[:-2, 2:] >= center).astype(np.uint8) << 2
        | (patch[1:-1, 2:] >= center).astype(np.uint8) << 3
        | (patch[2:, 2:] >= center).astype(np.uint8) << 4
        | (patch[2:, 1:-1] >= center).astype(np.uint8) << 5
        | (patch[2:, :-2] >= center).astype(np.uint8) << 6
        | (patch[1:-1, :-2] >= center).astype(np.uint8) << 7
    )
    return neighbors.ravel()


def compute_mel_spectrogram(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def extract_mel_spectrogram_features(
    mel_spectrogram_db: np.ndarray,
    cfg: FeatureConfig,
) -> dict[str, np.ndarray]:
    if not (cfg.include_mel_spectrogram or cfg.include_delta or cfg.include_delta2):
        return {}

    features: dict[str, np.ndarray] = {}
    if cfg.include_mel_spectrogram:
        features["mel_spectrogram"] = mel_spectrogram_db
    if cfg.include_delta:
        features["mel_spectrogram_delta"] = librosa.feature.delta(mel_spectrogram_db, order=1)
    if cfg.include_delta2:
        features["mel_spectrogram_delta2"] = librosa.feature.delta(mel_spectrogram_db, order=2)
    return features


def extract_spectral_features(
    mel_spectrogram_db: np.ndarray,
    cfg: FeatureConfig,
) -> dict[str, np.ndarray]:
    if not cfg.include_spectral:
        return {}

    mel_power = librosa.db_to_power(mel_spectrogram_db)
    mel_freqs = librosa.mel_frequencies(n_mels=mel_spectrogram_db.shape[0], fmax=cfg.sr / 2.0)
    mel_power_norm = mel_power / np.clip(np.sum(mel_power, axis=0, keepdims=True), 1e-12, None)

    spectral_entropy = -np.sum(mel_power_norm * _safe_log(mel_power_norm), axis=0, keepdims=True)
    spectral_flux = np.concatenate(
        [np.zeros((1, 1), dtype=float), np.sqrt(np.sum(np.square(np.diff(mel_power, axis=1)), axis=0, keepdims=True))],
        axis=1,
    )

    features: dict[str, np.ndarray] = {
        "spectral_centroid": librosa.feature.spectral_centroid(S=mel_power, freq=mel_freqs),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=mel_power, freq=mel_freqs),
        "spectral_rolloff": librosa.feature.spectral_rolloff(
            S=mel_power,
            freq=mel_freqs,
            roll_percent=cfg.roll_percent,
        ),
        "spectral_contrast": librosa.feature.spectral_contrast(
            S=mel_power,
            freq=mel_freqs,
            fmin=max(50.0, mel_freqs[1]),
        ),
        "spectral_entropy": spectral_entropy,
        "spectral_flux": spectral_flux,
    }
    return features


def extract_energy_features(
    mel_spectrogram_db: np.ndarray,
    cfg: FeatureConfig,
) -> dict[str, np.ndarray]:
    if not cfg.include_energy:
        return {}

    mel_power = librosa.db_to_power(mel_spectrogram_db)
    energy_proxy = np.mean(mel_power, axis=0)

    local_mean = _moving_average(energy_proxy, cfg.event_window_frames)
    local_std = _moving_std(energy_proxy, cfg.event_window_frames)
    burstiness = (local_std - local_mean) / np.clip(local_std + local_mean, 1e-12, None)

    threshold = float(np.mean(energy_proxy) + cfg.event_threshold_std * np.std(energy_proxy))
    is_peak = np.zeros_like(energy_proxy, dtype=bool)
    if energy_proxy.size > 0:
        is_peak[0] = energy_proxy[0] >= threshold
        is_peak[-1] = energy_proxy[-1] >= threshold
    if energy_proxy.size > 2:
        center = energy_proxy[1:-1]
        is_peak[1:-1] = (
            (center >= energy_proxy[:-2])
            & (center >= energy_proxy[2:])
            & (center >= threshold)
        )

    peak_idx = np.flatnonzero(is_peak)
    prev_gap = np.full(energy_proxy.shape[0], float(energy_proxy.shape[0]), dtype=float)
    next_gap = np.full(energy_proxy.shape[0], float(energy_proxy.shape[0]), dtype=float)
    if peak_idx.size > 0:
        pos = np.arange(energy_proxy.shape[0])
        prev_pos = np.searchsorted(peak_idx, pos, side="right") - 1
        next_pos = np.searchsorted(peak_idx, pos, side="left")

        valid_prev = prev_pos >= 0
        valid_next = next_pos < peak_idx.size
        prev_gap[valid_prev] = pos[valid_prev] - peak_idx[prev_pos[valid_prev]]
        next_gap[valid_next] = peak_idx[next_pos[valid_next]] - pos[valid_next]

    return {
        "spectral_energy_proxy": energy_proxy[np.newaxis, :],
        "burstiness": burstiness[np.newaxis, :],
        "inter_event_prev_gap": prev_gap[np.newaxis, :],
        "inter_event_next_gap": next_gap[np.newaxis, :],
    }


def extract_glcm_features(
    mel_spectrogram_db: np.ndarray,
    cfg: FeatureConfig,
) -> dict[str, np.ndarray]:
    if not cfg.include_texture:
        return {}

    mel_small = _downsample_freq_axis(mel_spectrogram_db, cfg.texture_downsample_freq_bins)
    quant = _quantize_spectrogram(mel_small, cfg.texture_quant_levels)
    n_frames = quant.shape[1]
    half_window = max(cfg.texture_patch_frames // 2, 1)

    feature_names = (
        "glcm_time_contrast",
        "glcm_time_entropy",
        "glcm_time_homogeneity",
        "glcm_time_energy",
        "glcm_freq_contrast",
        "glcm_freq_entropy",
        "glcm_freq_homogeneity",
        "glcm_freq_energy",
    )
    out = {name: np.zeros((1, n_frames), dtype=float) for name in feature_names}

    for frame_idx in range(n_frames):
        start = max(0, frame_idx - half_window)
        end = min(n_frames, frame_idx + half_window + 1)
        patch = quant[:, start:end]

        if patch.shape[1] >= 2:
            glcm_time = _glcm_from_pairs(patch[:, :-1], patch[:, 1:], cfg.texture_quant_levels)
            stats = _glcm_stats(glcm_time)
            out["glcm_time_contrast"][0, frame_idx] = stats[0]
            out["glcm_time_entropy"][0, frame_idx] = stats[1]
            out["glcm_time_homogeneity"][0, frame_idx] = stats[2]
            out["glcm_time_energy"][0, frame_idx] = stats[3]

        if patch.shape[0] >= 2:
            glcm_freq = _glcm_from_pairs(patch[:-1, :], patch[1:, :], cfg.texture_quant_levels)
            stats = _glcm_stats(glcm_freq)
            out["glcm_freq_contrast"][0, frame_idx] = stats[0]
            out["glcm_freq_entropy"][0, frame_idx] = stats[1]
            out["glcm_freq_homogeneity"][0, frame_idx] = stats[2]
            out["glcm_freq_energy"][0, frame_idx] = stats[3]

    return out


def extract_lbp_features(
    mel_spectrogram_db: np.ndarray,
    cfg: FeatureConfig,
) -> dict[str, np.ndarray]:
    if not cfg.include_texture:
        return {}

    mel_small = _downsample_freq_axis(mel_spectrogram_db, cfg.texture_downsample_freq_bins)
    quant = _quantize_spectrogram(mel_small, cfg.texture_quant_levels)
    n_frames = quant.shape[1]
    half_window = max(cfg.texture_patch_frames // 2, 1)

    lbp_hist = np.zeros((cfg.lbp_n_bins, n_frames), dtype=float)
    hist_edges = np.linspace(0, 256, cfg.lbp_n_bins + 1)

    for frame_idx in range(n_frames):
        start = max(0, frame_idx - half_window)
        end = min(n_frames, frame_idx + half_window + 1)
        patch = quant[:, start:end]
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            continue

        codes = _lbp_codes(patch)
        if codes.size == 0:
            continue
        hist, _ = np.histogram(codes, bins=hist_edges)
        lbp_hist[:, frame_idx] = hist / np.clip(np.sum(hist), 1.0, None)

    return {"lbp_hist": lbp_hist}


def extract_all_frame_features_from_mel(
    mel_spectrogram_db: np.ndarray,
    cfg: FeatureConfig,
) -> dict[str, np.ndarray]:
    mel_db = np.asarray(mel_spectrogram_db, dtype=float)
    if mel_db.ndim != 2:
        raise ValueError("mel_spectrogram_db must be 2D with shape (n_mels, n_frames)")

    features: dict[str, np.ndarray] = {}
    features.update(extract_mel_spectrogram_features(mel_db, cfg))
    features.update(extract_spectral_features(mel_db, cfg))
    features.update(extract_energy_features(mel_db, cfg))
    features.update(extract_glcm_features(mel_db, cfg))
    features.update(extract_lbp_features(mel_db, cfg))
    return features


def extract_all_frame_features(
    y: np.ndarray,
    cfg: FeatureConfig,
) -> dict[str, np.ndarray]:
    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.size == 0:
        raise ValueError("y must contain at least one sample")

    mel_spectrogram_db = compute_mel_spectrogram(y_arr, cfg)
    return extract_all_frame_features_from_mel(mel_spectrogram_db, cfg)


def compute_profile_similarity_features(
    X: np.ndarray,
    feature_names: list[str] | np.ndarray,
    species_profiles: np.ndarray,
    species_ids: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute cosine similarity between mel_spectrogram_mean features and class profiles.
    """
    return compute_profile_cosine_similarity(
        X=X,
        feature_names=feature_names,
        species_profiles=species_profiles,
        species_ids=species_ids,
        prefix="mel_spectrogram_",
        suffix="_mean",
    )


def compute_profile_similarity_from_mean_mel(
    mel_mean: np.ndarray,
    species_profiles: np.ndarray,
    species_ids: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute cosine similarity from preselected mel_spectrogram_mean vectors.
    """
    X = np.asarray(mel_mean, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]

    profiles = np.asarray(species_profiles, dtype=float)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    profiles_norm = np.linalg.norm(profiles, axis=1, keepdims=True).T
    denom = np.clip(X_norm * profiles_norm, 1e-12, None)
    similarities = (X @ profiles.T) / denom
    names = [f"cos_mel_spectrogram_mean_{species_id}" for species_id in species_ids]
    return similarities.astype(np.float32), names
