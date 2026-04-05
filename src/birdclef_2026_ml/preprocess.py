import ast
import numpy as np
import pandas as pd


def _safe_literal_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x


def preprocess_train(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()

    df["primary_label_list"] = df["primary_label"].str.split(";")
    df["primary_label_list"] = df["primary_label_list"].apply(
        lambda x: [s.strip() for s in x] if isinstance(x, list) else []
    )

    df[["date", "time"]] = df["filename"].str.extract(r'_(\d{8})_(\d{6})\.ogg$')
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y%m%d %H%M%S", errors="coerce")

    df.drop(columns=["date", "time"], inplace=True)
    return df
