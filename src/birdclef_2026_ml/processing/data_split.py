from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
# from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

# try:
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

from birdclef_2026_ml.paths import load_project_paths


def _to_label_list(label: Any) -> list[Any]:
    """Normalize a row label value to a list for multilabel grouping."""
    if isinstance(label, (str, bytes)):
        return [label]
    if isinstance(label, Iterable):
        return list(label)
    return [label]


def _build_grouped_filename_frame(
    filenames: np.ndarray,
    y_multilabel: np.ndarray | None = None,
) -> pd.DataFrame:
    split_df = pd.DataFrame(
        {
            "row_idx": np.arange(len(filenames), dtype=int),
            "filename": np.asarray(filenames),
        }
    )
    grouped = (
        split_df.groupby("filename", sort=False)
        .agg(row_idx=("row_idx", list))
        .reset_index(drop=False)
    )
    if y_multilabel is not None:
        grouped["labels"] = [
            np.asarray(y_multilabel[idx], dtype=int).max(axis=0)
            for idx in grouped["row_idx"]
        ]
    return grouped


def _expand_group_row_indices(grouped: pd.DataFrame, group_indices: np.ndarray) -> np.ndarray:
    return np.sort(np.concatenate(grouped.iloc[group_indices]["row_idx"].to_numpy()))


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
    """Split row indices for train/validation, grouped by filename.

    Assumes y is already MultiLabelBinarized (multi-hot per row).
    """
    n_samples = len(X)

    grouped = _build_grouped_filename_frame(filenames=np.asarray(filenames), y_multilabel=np.asarray(y))

    # Aggregate per-file multilabel targets via max over rows.
    y_grouped = np.vstack(grouped["labels"])
    x_grouped = grouped.index.to_numpy().reshape(-1, 1)

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=cast(Any, test_size),
        random_state=random_state,
    )
    group_train_idx, group_val_idx = next(splitter.split(x_grouped, cast(Any, y_grouped)))

    train_row_indices = _expand_group_row_indices(grouped, group_train_idx)
    val_row_indices = _expand_group_row_indices(grouped, group_val_idx)
    return train_row_indices, val_row_indices


def iter_soundscapes_oof_splits(
    y: np.ndarray,
    filenames: np.ndarray,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return grouped multilabel OOF splits for soundscape rows."""
    y = np.asarray(y, dtype=int)
    if y.ndim != 2:
        raise ValueError("y must be 2D multilabel matrix")
    if len(y) != len(filenames):
        raise ValueError("y and filenames must have same length")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    grouped = _build_grouped_filename_frame(filenames=np.asarray(filenames), y_multilabel=y)
    y_grouped = np.vstack(grouped["labels"])
    x_grouped = grouped.index.to_numpy().reshape(-1, 1)

    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for group_train_idx, group_val_idx in splitter.split(x_grouped, cast(Any, y_grouped)):
        train_row_indices = _expand_group_row_indices(grouped, np.asarray(group_train_idx, dtype=int))
        val_row_indices = _expand_group_row_indices(grouped, np.asarray(group_val_idx, dtype=int))
        splits.append((train_row_indices, val_row_indices))
    return splits


def get_idx_from_filenames(all_filenames):
    def get_idx(subset_filenames):
        return np.where(np.isin(all_filenames, subset_filenames))[0]

    path = load_project_paths().train_val_filenames
    train_filenames = np.load(path / "train_filenames.npy", allow_pickle=True)
    val_filenames = np.load(path / "val_filenames.npy", allow_pickle=True)

    train_idx = get_idx(train_filenames)
    val_idx = get_idx(val_filenames)
    return train_idx, val_idx
