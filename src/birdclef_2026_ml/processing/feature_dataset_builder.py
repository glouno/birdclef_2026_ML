import numpy as np
import pandas as pd
from pathlib import Path

from birdclef_2026_ml.configs import PipelineConfig
from birdclef_2026_ml.feature_engineering import (
    add_profile_similarity_features,
    extract_features_from_path,
)
from birdclef_2026_ml.feature_engineering.chunking import count_nb_chunks


def _reduce_mel_family_features(
    X: np.ndarray,
    feature_names: list[str],
    prefix: str,
    target_bins: int = 32,
) -> tuple[np.ndarray, list[str]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    keep_indices: list[int] = []
    keep_names: list[str] = []

    for idx, feature_name in enumerate(feature_names):
        name = str(feature_name)
        if not name.startswith(prefix):
            keep_indices.append(idx)
            keep_names.append(name)
            continue

        parts = name.split("_")
        if len(parts) < 4:
            keep_indices.append(idx)
            keep_names.append(name)
            continue

        dim = int(parts[-2])
        stat = parts[-1]
        family = "_".join(parts[:-2])
        grouped.setdefault(f"{family}|{stat}", []).append((dim, idx))

    reduced_cols: list[np.ndarray] = []
    reduced_names: list[str] = []
    if keep_indices:
        reduced_cols.append(X[:, keep_indices])
        reduced_names.extend(keep_names)

    for key in sorted(grouped.keys()):
        family, stat = key.split("|", 1)
        entries = sorted(grouped[key], key=lambda item: item[0])
        dim_groups = np.array_split(np.arange(len(entries)), target_bins)
        for out_dim, group in enumerate(dim_groups, start=1):
            if len(group) == 0:
                continue
            cols = [entries[i][1] for i in group]
            reduced_cols.append(np.median(X[:, cols], axis=1, keepdims=True))
            reduced_names.append(f"{family}_{out_dim:02d}_{stat}")

    X_reduced = np.hstack(reduced_cols) if reduced_cols else np.empty((X.shape[0], 0), dtype=float)
    return X_reduced, reduced_names


def reduce_feature_vector(
    X: np.ndarray,
    feature_names: list[str] | np.ndarray,
    target_mel_bins: int = 32,
) -> tuple[np.ndarray, list[str]]:
    """
    Reduce feature matrix:
    - drop *_std
    - collapse mel/mel_delta/mel_delta2 dims from 128 -> target_mel_bins with median
    """
    feature_names_list = [str(name) for name in feature_names]
    keep_mask = np.array([not name.endswith("_std") for name in feature_names_list], dtype=bool)
    X_reduced = np.asarray(X[:, keep_mask], dtype=np.float32)
    reduced_names = [name for name, keep in zip(feature_names_list, keep_mask) if keep]

    for prefix in ("mel_spectrogram_", "mel_spectrogram_delta_", "mel_spectrogram_delta2_"):
        X_reduced, reduced_names = _reduce_mel_family_features(
            X_reduced,
            reduced_names,
            prefix=prefix,
            target_bins=target_mel_bins,
        )

    return X_reduced, reduced_names


def reduce_feature_memmap(run_path: Path, target_mel_bins: int = 32) -> tuple[np.ndarray, list[str]]:
    """
    Reduce saved memmap artifacts.
    Saves reduced artifacts with `_reduced` suffix in same folder.
    """
    feature_names = np.load(run_path / "feature_names.npy", allow_pickle=True)
    X_shape = tuple(np.load(run_path / "shape_X.npy"))
    y_shape = tuple(np.load(run_path / "shape_y.npy"))
    X_dtype = np.load(run_path / "dtype_X.npy").item()
    y_dtype = np.load(run_path / "dtype_y.npy").item()

    X = np.memmap(run_path / "X.dat", dtype=X_dtype, mode="r", shape=(X_shape[0], X_shape[1]))
    y = np.memmap(run_path / "y.dat", dtype=y_dtype, mode="r", shape=y_shape)

    X_reduced, reduced_names = reduce_feature_vector(
        np.asarray(X, dtype=np.float32),
        feature_names,
        target_mel_bins=target_mel_bins,
    )

    X_out = np.memmap(
        run_path / "X_reduced.dat",
        mode="w+",
        dtype=np.float32,
        shape=X_reduced.shape,
    )
    X_out[:] = X_reduced
    X_out.flush()

    y_out = np.memmap(
        run_path / "y_reduced.dat",
        mode="w+",
        dtype=y_dtype,
        shape=y_shape,
    )
    y_out[:] = y
    y_out.flush()

    np.save(run_path / "feature_names_reduced.npy", np.asarray(reduced_names, dtype=object))
    np.save(run_path / "file_ids_reduced.npy", np.load(run_path / "file_ids.npy", allow_pickle=True))
    np.save(run_path / "shape_X_reduced.npy", np.array(X_reduced.shape))
    np.save(run_path / "shape_y_reduced.npy", np.array(y_shape))
    np.save(run_path / "dtype_X_reduced.npy", np.array(str(np.dtype(np.float32))))
    np.save(run_path / "dtype_y_reduced.npy", np.array(str(np.dtype(y_dtype))))

    return X_reduced, reduced_names


# TODO: adapt for soundscapes with start_sec, end_sec instead of custom chunking
# Build feature matrices with pooling
def build_memmap_from_chunks(
    df: pd.DataFrame,
    pipeline_cfg: PipelineConfig,
    pathroot: Path,
    filename_col: str,
    features_pathroot: Path,
    out_instances_path: Path,
    soundscapes: bool,
    profiles_path: Path | None = None,
    species_ids_path: Path | None = None,
    n_classes: int = 234
):

    #  Infer dimensions
    sample_matrix, base_feature_names = extract_features_from_path(
        df[filename_col].iloc[0],
        pipeline_cfg=pipeline_cfg,
        pathroot=pathroot,
        features_pathroot=features_pathroot
    )
    feature_names = list(base_feature_names)
    if profiles_path is not None or species_ids_path is not None:
        if profiles_path is None or species_ids_path is None:
            raise ValueError("profiles_path and species_ids_path must be provided together.")
        sample_matrix, feature_names = add_profile_similarity_features(
            np.asarray(sample_matrix, dtype=float),
            feature_names,
            profiles_path=profiles_path,
            species_ids_path=species_ids_path,
        )
    n_features = len(feature_names)
    n_instances = count_nb_chunks(df, pipeline_cfg.chunk, pipeline_cfg.feature.sr, pipeline_cfg.feature.hop_length)
    print("Total number of instances", n_instances)
    out_instances_path.mkdir(parents=True, exist_ok=True)
    X = np.memmap(
        out_instances_path / "X.dat",
        mode="w+",
        dtype=np.float32,
        shape=(n_instances, n_features)
    )

    if soundscapes:
        y = np.memmap(
            out_instances_path / "y.dat",
            mode="w+",
            dtype=np.int8,
            shape=(n_instances, 2, n_classes)  # [:, 0, :] primary label & [:, 1, :] class name
        )
    else:
        y = np.memmap(
            out_instances_path / "y.dat",
            mode="w+",
            dtype=np.int32,
            shape=(n_instances, 2)
        )

    # Fill memmaps
    file_ids = np.empty(n_instances, dtype=object)
    cursor = 0
    for row in df.itertuples(index=False):
        print(f"Cursor {cursor}/{n_instances}")
        filename = getattr(row, filename_col)

        matrix, _ = extract_features_from_path(
            getattr(row, filename_col),
            pipeline_cfg,
            pathroot,
            features_pathroot=features_pathroot
        )
        matrix = np.asarray(matrix, dtype=float)
        if profiles_path is not None or species_ids_path is not None:
            matrix, _ = add_profile_similarity_features(
                matrix,
                list(base_feature_names),
                profiles_path=profiles_path,
                species_ids_path=species_ids_path,
            )

        n_chunks = len(matrix)
        X[cursor:cursor+n_chunks, :] = np.asarray(matrix, dtype=np.float32)
        if soundscapes:
            pl_indices = getattr(row, "primary_label_int_list")
            cn_indices = getattr(row, "class_sname_int_list")

            vec_pl = np.zeros(n_classes, dtype=np.int8)
            vec_cn = np.zeros(n_classes, dtype=np.int8)

            vec_pl[pl_indices] = 1
            vec_cn[cn_indices] = 1

            y[cursor:cursor+n_chunks, 0, :] = vec_cn
            y[cursor:cursor+n_chunks, 1, :] = vec_pl

        else:
            pl = getattr(row, "primary_label_int")
            cn = getattr(row, "class_name_int")

            y[cursor:cursor+n_chunks, 0] = cn
            y[cursor:cursor+n_chunks, 1] = pl

        file_ids[cursor:cursor+n_chunks] = filename
        cursor += n_chunks

    X.flush()
    y.flush()

    # Metadata
    np.save(out_instances_path / "feature_names.npy", feature_names)
    np.save(out_instances_path / "file_ids.npy", file_ids)
    np.save(out_instances_path / "shape_X.npy", np.array(X.shape))
    np.save(out_instances_path / "shape_y.npy", np.array(y.shape))
    np.save(out_instances_path / "dtype_X.npy", np.array(str(X.dtype)))
    np.save(out_instances_path / "dtype_y.npy", np.array(str(y.dtype)))
