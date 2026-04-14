import librosa
from pathlib import Path
import pandas as pd
import soundfile as sf

from birdclef_2026_ml.constants import SAMPLE_RATE
from birdclef_2026_ml.paths import PATHS


def load_audio(filepath):
    y, _ = librosa.load(filepath, sr=SAMPLE_RATE)
    return y


def load_train_audio(train_df, idx=0):
    filename = train_df["filename"].iloc[idx]
    audio_path = Path(PATHS["train_audio_dir"]) / filename
    y = load_audio(audio_path)
    return y, audio_path


def get_duration(filepath):
    info = sf.info(PATHS["train_audio_dir"] / filepath)
    return float(info.duration)
