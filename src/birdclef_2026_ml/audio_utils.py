import librosa
from pathlib import Path
import soundfile as sf

from birdclef_2026_ml.constants import SAMPLE_RATE
from birdclef_2026_ml.paths import PATHS


def load_audio(filepath: str | Path, sr: int = SAMPLE_RATE):
    y, _ = librosa.load(filepath, sr=sr)
    return y


def get_path(pathroot: str, filename: str | Path | None = None) -> Path:
    root = Path(PATHS[pathroot])
    if filename is None:
        return root
    return root / Path(filename)


def build_audio_path(
    df,
    idx: int,
    pathroot: str,
    filename_col: str = "filename",
) -> Path:
    filename = df[filename_col].iloc[idx]
    return get_path(pathroot, filename)


def build_audio_path_row(
    df,
    idx: int,
    pathroot: str,
    filename_col: str = "filename",
) -> Path:
    return build_audio_path(df=df, idx=idx, pathroot=pathroot, filename_col=filename_col)


def load_train_audio(
    df,
    idx: int,
    pathroot: str = "train_audio_dir",
    filename_col: str = "filename",
    sr: int = SAMPLE_RATE,
):
    audio_path = build_audio_path(
        df=df,
        idx=idx,
        pathroot=pathroot,
        filename_col=filename_col,
    )
    y = load_audio(audio_path, sr=sr)
    return y, audio_path


def get_duration(filepath: str | Path, pathroot: str = "train_audio_dir"):
    filepath = Path(filepath)
    audio_path = filepath if filepath.is_absolute() else get_path(pathroot, filepath)
    info = sf.info(audio_path)
    return float(info.duration)
