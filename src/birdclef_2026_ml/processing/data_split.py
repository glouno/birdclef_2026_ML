from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Any, cast

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import pandas as pd


def split_audio_train_val(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/validation, put rare (appear only once) labels into train"""

    vc = df["primary_label"].value_counts()
    valid = vc[vc >= 2].index

    train_valid = df[df["primary_label"].isin(valid)]
    rare = df[~df["primary_label"].isin(valid)]

    train_part, val_part = train_test_split(
        train_valid,
        stratify=train_valid["primary_label"],
        test_size=0.2,
        random_state=42
    )

    train_final = pd.concat([train_part, rare])
    return train_final, val_part


def split_soundscapes_train_val(soundscapes, test_size=0.2, random_state=42):
    """Split soundscapes into train/validation with multilabel stratification."""

    if "primary_label_list" in soundscapes.columns:
        label_lists = soundscapes["primary_label_list"].apply(
            lambda x: x if isinstance(x, list) else []
        )
    elif "primary_label" in soundscapes.columns:
        label_lists = soundscapes["primary_label"].fillna("").apply(
            lambda x: [lbl.strip() for lbl in str(x).split(";") if lbl.strip()]
        )
    else:
        raise ValueError("soundscapes must include 'primary_label' or 'primary_label_list'")

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(label_lists)

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=cast(Any, test_size),
        random_state=random_state,
    )
    x_dummy = soundscapes.index.to_numpy().reshape(-1, 1)
    train_idx, val_idx = next(splitter.split(x_dummy, cast(Any, y)))

    train_part = soundscapes.iloc[train_idx]
    val_part = soundscapes.iloc[val_idx]
    return train_part, val_part
