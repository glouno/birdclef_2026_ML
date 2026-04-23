from pydub import AudioSegment
import numpy as np
from pathlib import Path
import json

import librosa
import soundfile as sf

from birdclef_2026_ml.constants import SAMPLE_RATE
from birdclef_2026_ml.paths import PATHS


def load_audio(
    filepath: str | Path,
    sr: int = SAMPLE_RATE,
    offset: float = 0.0,
    duration: float | None = None,
):
    y, _ = librosa.load(filepath, sr=sr, offset=offset, duration=duration)
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
    row,
    pathroot: str,
    filename_col: str = "filename",
) -> Path:
    return get_path(pathroot, filename=row[filename_col])


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
    y = load_audio(audio_path, sr)
    return y, audio_path


def load_soundscape_audio(
    soundscapes,
    idx: int,
    pathroot: str = "train_soundscapes_dir",
    filename_col: str = "filename",
    start_col: str = "start_sec",
    end_col: str = "end_sec",
    sr: int = SAMPLE_RATE,
    load_full: bool = False
):
    row = soundscapes.iloc[idx]
    audio_path = get_path(pathroot, row[filename_col])

    if load_full:
        y = load_audio(filepath=audio_path, sr=sr)
        return y, audio_path

    offset_seconds = row[start_col]
    end_seconds = row[end_col]
    duration_seconds = end_seconds - offset_seconds

    y = load_audio(
        filepath=audio_path,
        sr=sr,
        offset=offset_seconds,
        duration=duration_seconds,
    )
    return y, audio_path, offset_seconds, end_seconds


def get_duration(filepath: str | Path, pathroot: str = "train_audio_dir"):
    filepath = Path(filepath)
    audio_path = filepath if filepath.is_absolute() else get_path(pathroot, filepath)
    info = sf.info(audio_path)
    return float(info.duration)


def save_ogg(path, y, sr):
    # Cconvert to int16 (pydub works with integer PCM internally)
    y_int16 = (y * 32767).astype(np.int16)

    audio = AudioSegment(
        y_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,  # int16 = 2 bytes
        channels=1
    )
    audio.export(str(path), format="ogg", codec="libvorbis")


def load_config(path: Path):
    config = dict()
    with open(path) as f:
        config = json.load(f)
    return config
