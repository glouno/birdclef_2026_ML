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


def load_memmap_dataset(run_name: str) -> MemmapDataset:
    run_path = load_project_paths().run_dir(run_name) / "data"

    filenames = np.load(run_path / "file_ids.npy", allow_pickle=True)
    feature_names = np.load(run_path / "feature_names.npy", allow_pickle=True)
    y_shape = tuple(np.load(run_path / "shape_y.npy"))
    X_shape = tuple(np.load(run_path / "shape_X.npy"))
    y_dtype = np.load(run_path / "dtype_y.npy").item()
    X_dtype = np.load(run_path / "dtype_X.npy").item()

    X = np.memmap(run_path / "X.dat", dtype=X_dtype, mode="r", shape=(X_shape[0], X_shape[1]))
    y = np.memmap(run_path / "y.dat", dtype=y_dtype, mode="r", shape=y_shape)

    return MemmapDataset(
        run_name=run_name,
        filenames=filenames,
        feature_names=feature_names,
        X=X,
        y=y,
    )
