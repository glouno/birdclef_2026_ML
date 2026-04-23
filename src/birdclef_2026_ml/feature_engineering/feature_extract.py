import numpy as np
import librosa

from birdclef_2026_ml.feature_engineering.configs import FeatureConfig


def extract_mel_spectrogram_features(y: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    """Extract frame-level mel spectrogram features for later class-specific weighting."""
    if not cfg.include_mel_spectrogram:
        return {}

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return {"mel_spectrogram": mel_spectrogram_db}


def extract_mfcc_features(y: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    """Extract MFCC frame features and optional deltas from a waveform.

    Returns keys among: `mfcc`, `mfcc_delta`, `mfcc_delta2` depending on
    `FeatureConfig.include_*` flags.
    """
    if not (cfg.include_mfcc or cfg.include_delta or cfg.include_delta2):
        return {}

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )

    features: dict[str, np.ndarray] = {}
    if cfg.include_mfcc:
        features["mfcc"] = mfcc
    if cfg.include_delta:
        features["mfcc_delta"] = librosa.feature.delta(mfcc, order=1)
    if cfg.include_delta2:
        features["mfcc_delta2"] = librosa.feature.delta(mfcc, order=2)

    return features


def extract_spectral_features(y: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    """Extract frame-level spectral features from a waveform."""
    if not cfg.include_spectral:
        return {}

    features: dict[str, np.ndarray] = {
        "spectral_centroid": librosa.feature.spectral_centroid(
            y=y,
            sr=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(
            y=y,
            sr=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
        "spectral_rolloff": librosa.feature.spectral_rolloff(
            y=y,
            sr=cfg.sr,
            roll_percent=cfg.roll_percent,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
        "spectral_contrast": librosa.feature.spectral_contrast(
            y=y,
            sr=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
    }

    return features


def extract_energy_features(y: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    """Extract frame-level energy features from a waveform."""
    if not cfg.include_energy:
        return {}

    features: dict[str, np.ndarray] = {
        "rms": librosa.feature.rms(
            y=y,
            frame_length=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
        "zcr": librosa.feature.zero_crossing_rate(
            y,
            frame_length=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
    }

    return features


def extract_all_frame_features(
    y: np.ndarray,
    cfg: FeatureConfig,
) -> dict[str, np.ndarray]:
    """Extract all enabled frame-level features in deterministic order."""
    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.size == 0:
        raise ValueError("y must contain at least one sample")

    features: dict[str, np.ndarray] = {}

    if cfg.include_waveform_stats:
        features["waveform"] = y_arr[np.newaxis, :]

    features.update(extract_mel_spectrogram_features(y_arr, cfg))
    features.update(extract_mfcc_features(y_arr, cfg))
    features.update(extract_spectral_features(y_arr, cfg))
    features.update(extract_energy_features(y_arr, cfg))
    return features
