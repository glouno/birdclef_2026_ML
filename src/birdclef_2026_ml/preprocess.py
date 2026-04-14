import ast
import numpy as np
import pandas as pd

from birdclef_2026_ml.audio_utils import get_duration


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


def preprocess_soundscape(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess before EDA"""

    df = df.copy()

    df["primary_label_list"] = df["primary_label"].str.split(";")
    df["primary_label_list"] = df["primary_label_list"].apply(
        lambda x: [s.strip() for s in x] if isinstance(x, list) else []
    )

    # Datetime extraction
    df[["date", "time"]] = df["filename"].str.extract(r'_(\d{8})_(\d{6})\.ogg$')
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y%m%d %H%M%S", errors="coerce")

    df.drop(columns=["date", "time"], inplace=True)
    return df


def preprocess_train_for_models(df: pd.DataFrame,
                                trim_percentile: float = 5.0) -> pd.DataFrame:
    """Preprocessing after EDA"""

    df = preprocess_train(df)

    # Remove audios that are too short/too long
    df["duration"] = df["filename"].apply(get_duration)
    p5, p95 = np.percentile(df["duration"], [trim_percentile, 100 - trim_percentile])
    df = df[(df["duration"] > p5) & (df["duration"] < p95)]

    return df
