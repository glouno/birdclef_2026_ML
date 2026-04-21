import librosa
from pathlib import Path
import soundfile as sf
import numpy as np
import pandas as pd

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
):
    row = soundscapes.iloc[idx]
    audio_path = get_path(pathroot, row[filename_col])

    offset_seconds = row[start_col]
    end_seconds = row[end_col]
    duration_seconds = end_seconds - offset_seconds

    y, _ = librosa.load(
        audio_path,
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


def compute_rms_dbfs(y, sr, frame_length, hop_length, ref_for_db):
    # Convert to dBFS (decibels relative to full scale)
    y = y / np.max(y)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    db = librosa.amplitude_to_db(rms, ref=ref_for_db)
    times = times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    return rms, db, times


def select_silence_frames_from_rms_db(
    rms_db,
    times,
    silence_th=-40.0,
    start_th=None,
    stop_th=None,
):
    """Return silence mask and silence-frame seconds from RMS dB.

    If start_th and stop_th are provided, hysteresis is used:
    - enter silence when rms_db <= start_th
    - leave silence when rms_db >= stop_th
    """
    rms_db = np.asarray(rms_db, dtype=float)
    times = np.asarray(times, dtype=float)

    if start_th is None or stop_th is None:
        silent_mask = rms_db <= silence_th
    else:
        silent_mask = np.zeros_like(rms_db, dtype=bool)
        in_silence = rms_db[0] <= start_th
        for i, db in enumerate(rms_db):
            if in_silence and db >= stop_th:
                in_silence = False
            elif (not in_silence) and db <= start_th:
                in_silence = True
            silent_mask[i] = in_silence

    silent_segments = _silence_segments_from_mask(silent_mask, times)
    return silent_mask, times[silent_mask], silent_segments


def _silence_segments_from_mask(silent_mask, times):
    """Convert a boolean silence mask into contiguous (start_s, end_s) segments."""
    silent_mask = np.asarray(silent_mask, dtype=bool)
    times = np.asarray(times, dtype=float)

    if len(times) == 0:
        return []

    dt = np.median(np.diff(times)) if len(times) > 1 else 0.0
    padded = np.r_[False, silent_mask, False]
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1

    return [(times[s], times[e] + dt) for s, e in zip(starts, ends)]
