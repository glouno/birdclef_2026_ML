import json
import numpy as np
import librosa

from birdclef_2026_ml.paths import get_path
from birdclef_2026_ml.processing.audio_utils import load_audio, build_audio_path, save_ogg
# Import SpectralGatingConfig
from birdclef_2026_ml.audio.spectral_gating import spectral_gating_snr
from birdclef_2026_ml.feature_engineering.configs import SpectralGatingConfig
from birdclef_2026_ml.audio.freq_bands import weighted_time_pooling


def apply_spectral_gating(
    df,
    config: SpectralGatingConfig,
    input_root: str = "train_audio_dir",
    output_root: str = "train_audio_spectral_gating_dir",
    filename_col: str = "filename",
):
    """
    Apply spectral gating to all audio files in df and save cleaned audio to output_root.
    Also exports a config.json with the parameters used for spectral gating.
    """

    for idx in df.index:
        input_path = build_audio_path(
            df=df,
            idx=idx,
            pathroot=input_root,
            filename_col=filename_col,
        )

        y = load_audio(input_path, sr=config.sr)
        y_clean, _, _, _ = spectral_gating_snr(y, config)

        output_path = build_audio_path(
            df=df,
            idx=idx,
            pathroot=output_root,
            filename_col=filename_col,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_ogg(output_path, y_clean, config.sr)


def build_profile(
    y_clean,
    sr,
    n_mels=128,
    n_fft=1024,
    hop_length=512,
    pooling_method="percentile",
    percentile=80,
    **kwargs
):
    """
    Compute a mel spectrogram profile from cleaned audio using weighted_time_pooling.

    Parameters:
        y_clean: np.ndarray
            Cleaned audio signal.
        sr: int
            Sample rate.
        n_mels: int, optional
            Number of mel bands to generate.
        n_fft: int, optional
            FFT window size.
        hop_length: int, optional
            Number of samples between successive frames.
        pooling_method: str, optional
            Pooling method for weighted_time_pooling (e.g., 'percentile', 'mean', etc.).
        percentile: float, optional
            Percentile value if method is 'percentile'.
        **kwargs: dict
            Additional arguments for weighted_time_pooling.

    Returns:
        profile: np.ndarray
            The pooled profile vector.
    """
    S = librosa.feature.melspectrogram(y=y_clean, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    profile = weighted_time_pooling(
        S_db,
        method=pooling_method,
        percentile=percentile,
        **kwargs
    )
    return profile
