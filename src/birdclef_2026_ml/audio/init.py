from pathlib import Path

from birdclef_2026_ml.processing.audio_utils import load_audio, build_audio_path, save_ogg
from birdclef_2026_ml.audio.spectral_gating import spectral_gating_snr
from birdclef_2026_ml.configs import SpectralGatingConfig


def apply_spectral_gating(
    df,
    config: SpectralGatingConfig,
    input_root: Path,
    output_root: Path,
    filename_col: str = "filename",
):
    """
    Apply spectral gating to all audio files in df and save cleaned audio to output_root.
    Also exports a config.json with params used for spectral gating.
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
