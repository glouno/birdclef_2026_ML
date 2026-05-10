import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from birdclef_2026_ml.paths import load_project_paths


@dataclass(frozen=True)
class MemmapDataset:
    run_name: str
    filenames: np.ndarray
    feature_names: np.ndarray
    X: np.memmap
    y: np.memmap
    bags_meta: np.ndarray | None


def _read_metadata_json(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_memmap_dataset(run_name: str, reduced: bool = True, soundscape: bool = True) -> MemmapDataset:
    run_path = load_project_paths().run_dir(run_name) / "data"
    run_path = run_path / "full" if not reduced else run_path / "reduced"

    suffix = "_soundscape" if soundscape else ""
    metadata_path = run_path / f"metadata{suffix}.json"
    metadata = _read_metadata_json(metadata_path)

    filenames = None
    feature_names = None
    X_shape = None
    y_shape = None
    X_dtype = None
    y_dtype = None
    X_path = run_path / f"X{suffix}.dat"
    y_path = run_path / f"y{suffix}.dat"
    bags_meta = None

    x_meta = metadata.get("X")
    y_meta = metadata.get("y")
    if x_meta is None or y_meta is None:
        raise ValueError(f"Missing X/y metadata in {metadata_path}")

    X_shape = tuple(int(value) for value in x_meta["shape"])
    y_shape = tuple(int(value) for value in y_meta["shape"])
    X_dtype = np.dtype(x_meta["dtype"])
    y_dtype = np.dtype(y_meta["dtype"])
    X_path = run_path / str(x_meta.get("path", f"X{suffix}.dat"))
    y_path = run_path / str(y_meta.get("path", f"y{suffix}.dat"))

    # if "feature_names" in metadata:
    feature_names = np.asarray(metadata["feature_names"], dtype=object)
    # if "file_ids" in metadata:
    filenames = np.asarray(metadata["file_ids"], dtype=object)

    bags_meta_entry = metadata.get("bags_meta")
    if isinstance(bags_meta_entry, dict):
        bags_meta_path = run_path / str(bags_meta_entry.get("path", f"bags_meta{suffix}.npy"))
        if bags_meta_path.exists():
            bags_meta = np.load(bags_meta_path, allow_pickle=True)

    # if feature_names is None:
    #     feature_names_path = run_path / f"feature_names{suffix}.npy"
    #     feature_names = (
    #         np.load(feature_names_path, allow_pickle=True)
    #         if feature_names_path.exists()
    #         else np.array([], dtype=object)
    #     )

    # if filenames is None:
    #     filenames_path = run_path / f"file_ids{suffix}.npy"
    #     filenames = (
    #         np.load(filenames_path, allow_pickle=True)
    #         if filenames_path.exists()
    #         else np.array([], dtype=object)
    #     )

    # if bags_meta is None:
    #     bags_meta_path = run_path / f"bags_meta{suffix}.npy"
    #     if bags_meta_path.exists():
    #         bags_meta = np.load(bags_meta_path, allow_pickle=True)

    X = np.memmap(X_path, dtype=X_dtype, mode="r", shape=X_shape)
    y = np.memmap(y_path, dtype=y_dtype, mode="r", shape=y_shape)

    return MemmapDataset(
        run_name=run_name,
        filenames=filenames,
        feature_names=feature_names,
        X=X,
        y=y,
        bags_meta=bags_meta,
    )


@dataclass(frozen=True)
class OVRInferenceMemmap:
    class_name: np.memmap | None = None
    primary_label: np.memmap | None = None
    proba: np.memmap | None = None


def save_memmap_array(inference_dir: Path, name: str, array: np.ndarray) -> Path:
    array = np.asarray(array)
    memmap_path = inference_dir / f"{name}.dat"
    metadata_path = inference_dir / "metadata.json"

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    arrays_meta = metadata.get("arrays", {})
    arrays_meta[name] = {
        "shape": [int(value) for value in array.shape],
        "dtype": np.dtype(array.dtype).str,
        "path": memmap_path.name,
    }
    metadata["arrays"] = arrays_meta

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    memmap_arr = np.memmap(memmap_path, dtype=array.dtype, mode="w+", shape=array.shape)
    memmap_arr[:] = array
    memmap_arr.flush()
    return memmap_path


def load_memmap_array(inference_dir: Path, name: str) -> np.memmap:
    memmap_path = inference_dir / f"{name}.dat"

    metadata_path = inference_dir / "metadata.json"
    metadata = _read_metadata_json(metadata_path)

    arrays_meta = metadata.get("arrays", {})
    # if name in arrays_meta:
    entry = arrays_meta[name]
    shape = tuple(int(value) for value in entry["shape"])
    dtype = np.dtype(entry["dtype"])
    return np.memmap(memmap_path, dtype=dtype, mode="r", shape=shape)


def load_ovr_inference_memmap(
    inference_dir: Path,
    output_stem: str,
    *,
    dual: bool,
) -> OVRInferenceMemmap:
    if dual:
        class_name = load_memmap_array(
            inference_dir, f"{output_stem}_class_name_probas"
        )
        primary_label = load_memmap_array(
            inference_dir, f"{output_stem}_primary_label_probas"
        )
        return OVRInferenceMemmap(
            class_name=class_name,
            primary_label=primary_label,
        )

    proba = load_memmap_array(inference_dir, f"{output_stem}_probas")
    return OVRInferenceMemmap(proba=proba)
