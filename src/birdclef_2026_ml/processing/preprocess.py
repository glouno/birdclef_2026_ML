import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from birdclef_2026_ml.processing.audio_utils import get_duration


def _safe_literal_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x


def preprocess_train(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess before EDA"""
    df = df.copy()
    df = df.drop_duplicates()

    # Parse stringified lists
    df["secondary_labels"] = df["secondary_labels"].apply(_safe_literal_eval)
    df["type"] = df["type"].apply(_safe_literal_eval)

    # Handle missing / placeholder values
    df["rating"] = df["rating"].replace(0, np.nan)
    df["author"] = df["author"].replace("Unknown", np.nan)

    # Normalize empty lists instead of NaN for consistency
    df["secondary_labels"] = df["secondary_labels"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df["type"] = df["type"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return df


def _series_to_seconds(values: pd.Series) -> pd.Series:
    """Convert a Series of numeric/timedelta-like values to seconds."""
    seconds = pd.to_timedelta(values.astype("string"), errors="coerce").dt.total_seconds()

    return seconds.astype(float)


def _add_soundscape_time_columns(df: pd.DataFrame):
    """Add start/end time in seconds and validate positive segment durations."""
    df["start_sec"] = _series_to_seconds(df["start"])
    df["end_sec"] = _series_to_seconds(df["end"])


def preprocess_soundscape(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess before EDA"""

    df = df.copy()
    df = df.drop_duplicates()

    df["primary_label_list"] = df["primary_label"].str.split(";")
    df["primary_label_list"] = df["primary_label_list"].apply(
        lambda x: [s.strip() for s in x] if isinstance(x, list) else []
    )

    # Datetime extraction
    df[["date", "time"]] = df["filename"].str.extract(r'_(\d{8})_(\d{6})\.ogg$')
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y%m%d %H%M%S", errors="coerce")

    df.drop(columns=["date", "time"], inplace=True)
    _add_soundscape_time_columns(df)

    return df


def preprocess_datasets_for_models(train: pd.DataFrame,
                                   soundscapes: pd.DataFrame,
                                   taxonomy: pd.DataFrame):

    # 1. Clean train audio
    train = preprocess_train(train)

    # Remove audios that are too short/too long
    # train["duration"] = train["filename"].apply(get_duration)
    # p5, p95 = np.percentile(train["duration"], [5., 95.])
    # train = train[(train["duration"] > p5) & (train["duration"] < p95)]

    # String labels to int
    le_primary_label = LabelEncoder().fit(taxonomy["primary_label"])
    le_class_name = LabelEncoder().fit(taxonomy["class_name"])

    train = train.assign(
        primary_label_int=np.asarray(
            le_primary_label.transform(train["primary_label"]),
            dtype=np.float32
        ),
        class_name_int=np.asarray(
            le_class_name.transform(train["class_name"]),
            dtype=np.float32
        )
    )

    # 2. Soundscapes
    soundscapes = preprocess_soundscape(soundscapes)

    # map primary_label -> class_name
    class_map = taxonomy.set_index("primary_label")["class_name"]
    exploded = soundscapes["primary_label_list"].explode()
    class_name_list = (
        exploded
        .map(class_map)
        .groupby(level=0)
        .agg(list)
    )

    soundscapes = soundscapes.assign(
        class_name_list=class_name_list
    )

    # primary_label_int_list
    soundscapes = soundscapes.assign(
        primary_label_int_list=soundscapes["primary_label_list"].apply(
            lambda x: np.asarray(le_primary_label.transform(x), dtype=np.float32).tolist()
        )
    )

    # class_name_int_list
    soundscapes = soundscapes.assign(
        class_name_int_list=soundscapes["class_name_list"].apply(
            lambda x: np.asarray(np.unique(le_class_name.transform(x)), dtype=np.float32).tolist()
        )
    )

    return train, soundscapes, le_primary_label, le_class_name
