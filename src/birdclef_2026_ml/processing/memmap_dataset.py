from dataclasses import dataclass

import numpy as np

from birdclef_2026_ml.paths import load_project_paths


@dataclass(frozen=True)
class MemmapDataset:
    run_name: str
    filenames: np.ndarray
    feature_names: np.ndarray
    X: np.memmap
    y: np.memmap


def load_memmap_dataset(run_name: str, reduced: bool = True) -> MemmapDataset:
    run_path = load_project_paths().run_dir(run_name) / "data"
    suffix = "_reduced" if reduced else ""
    filenames = np.load(run_path / f"file_ids{suffix}.npy", allow_pickle=True)
    feature_names = np.load(run_path / f"feature_names{suffix}.npy", allow_pickle=True)
    y_shape = tuple(np.load(run_path / f"shape_y{suffix}.npy"))
    X_shape = tuple(np.load(run_path / f"shape_X{suffix}.npy"))
    y_dtype = np.load(run_path / f"dtype_y{suffix}.npy").item()
    X_dtype = np.load(run_path / f"dtype_X{suffix}.npy").item()

    X = np.memmap(run_path / "X{suffix}.dat", dtype=X_dtype, mode="r", shape=(X_shape[0], X_shape[1]))
    y = np.memmap(run_path / "y{suffix}.dat", dtype=y_dtype, mode="r", shape=y_shape)

    return MemmapDataset(
        run_name=run_name,
        filenames=filenames,
        feature_names=feature_names,
        X=X,
        y=y,
    )
