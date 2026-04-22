import json
import numpy as np


from birdclef_2026_ml.paths import get_path
from birdclef_2026_ml.processing.audio_utils import load_audio, build_audio_path, save_ogg
from birdclef_2026_ml.audio.spectral_gating import spectral_gating_snr


def apply_spectral_gating(df,
                          sr: int,
                          n_fft: int,
                          hop_length: int,
                          noise_percentile: float,
                          alpha: float,
                          smooth_freq: int = 3,
                          smooth_time: int = 3,
                          eps: float = 1e-8,
                          input_root: str = "train_audio_dir",
                          output_root: str = "train_audio_spectral_gating_dir",
                          filename_col: str = "filename",
                          ):
    """
    Apply spectral gating to all audio files in df and save cleaned audio to output_root.
    Also exports a config.json with the parameters used for spectral gating.
    """

    # Export config
    config = {
        "sr": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "noise_percentile": noise_percentile,
        "alpha": alpha,
        "smooth_freq": smooth_freq,
        "smooth_time": smooth_time,
        "eps": eps,
    }

    output_root_path = get_path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    config_path = output_root_path / "config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    for idx in df.index:
        input_path = build_audio_path(
            df=df,
            idx=idx,
            pathroot=input_root,
            filename_col=filename_col,
        )

        y = load_audio(input_path, sr=sr)
        y_clean, _, _, _ = spectral_gating_snr(
            y, n_fft, hop_length, noise_percentile, alpha, smooth_freq, smooth_time, eps)

        output_path = build_audio_path(
            df=df,
            idx=idx,
            pathroot=output_root,
            filename_col=filename_col,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_ogg(output_path, y_clean, sr)
