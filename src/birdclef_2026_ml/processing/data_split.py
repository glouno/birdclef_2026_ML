from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
# from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

# try:
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
# except ModuleNotFoundError:  # pragma: no cover - optional dependency
#     MultilabelStratifiedShuffleSplit = None


def _to_label_list(label: Any) -> list[Any]:
    """Normalize a row label value to a list for multilabel grouping."""
    if isinstance(label, (str, bytes)):
        return [label]
    if isinstance(label, Iterable):
        return list(label)
    return [label]


def _split_train_val_grouped_by_filename(
    X: np.ndarray,
    y: np.ndarray,
    filenames: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row indices for train/validation with grouping by filename."""
    n_samples = len(X)

    split_df = pd.DataFrame(
        {
            "row_idx": np.arange(n_samples, dtype=int),
            "filename": np.asarray(filenames),
            "labels": [_to_label_list(label) for label in y],
        }
    )

    grouped = (
        split_df.groupby("filename", sort=False)
        .agg(
            row_idx=("row_idx", list),
            labels=("labels", "first"),
        )
        .reset_index(drop=False)
    )

    mlb = MultiLabelBinarizer()
    y_grouped = mlb.fit_transform(grouped["labels"])
    x_grouped = grouped.index.to_numpy().reshape(-1, 1)

    # try:
    #     if MultilabelStratifiedShuffleSplit is None:
    #         raise ValueError("iterstrat is not installed")
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=cast(Any, test_size),
        random_state=random_state,
    )
    group_train_idx, group_val_idx = next(splitter.split(x_grouped, cast(Any, y_grouped)))
    # except ValueError:
    #     # Fallback when stratified multilabel constraints cannot be satisfied
    #     # or optional dependency `iterstrat` is unavailable.
    #     splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    #     group_ids = np.arange(len(grouped))
    #     group_train_idx, group_val_idx = next(splitter.split(group_ids, groups=group_ids))

    train_row_indices = np.sort(
        np.concatenate(grouped.iloc[group_train_idx]["row_idx"].to_numpy())
    )
    val_row_indices = np.sort(
        np.concatenate(grouped.iloc[group_val_idx]["row_idx"].to_numpy())
    )
    return train_row_indices, val_row_indices


def split_audio_train_val(
    X: np.ndarray,
    y: np.ndarray,
    filenames: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices for train/validation, grouped by filename."""
    return _split_train_val_grouped_by_filename(
        X=X,
        y=y,
        filenames=filenames,
        test_size=test_size,
        random_state=random_state,
    )


def split_soundscapes_train_val(
    X: np.ndarray,
    y: np.ndarray,
    filenames: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices for train/validation, grouped by filename."""
    return _split_train_val_grouped_by_filename(
        X=X,
        y=y,
        filenames=filenames,
        test_size=test_size,
        random_state=random_state,
    )
