import numpy as np
import pandas as pd
from pathlib import Path

from birdclef_2026_ml.configs import ChunkConfig, PipelineConfig
from birdclef_2026_ml.feature_engineering import (
    add_profile_similarity_features,
    build_feature_vector,
    extract_features_from_path,
)
from birdclef_2026_ml.feature_engineering.chunking import (
    count_nb_chunks,
    get_duration_chunk_intervals,
)
from birdclef_2026_ml.processing.audio_utils import load_audio


def _slice_mel_spectrogram_by_seconds(
    mel_spectrogram: np.ndarray,
    start_sec: float,
    end_sec: float,
    sr: int,
    hop_length: int,
) -> np.ndarray:
    mel = np.asarray(mel_spectrogram, dtype=float)
    if mel.ndim != 2:
        raise ValueError("mel_spectrogram must be 2D")

    start_frame = max(0, int(np.floor(float(start_sec) * sr / hop_length)))
    end_frame = max(start_frame + 1, int(np.ceil(float(end_sec) * sr / hop_length)))
    end_frame = min(end_frame, mel.shape[1])
    if end_frame <= start_frame:
        return mel[:, start_frame:start_frame]
    return mel[:, start_frame:end_frame]


def _extract_soundscape_segment_features(
    filename: str,
    start_sec: float,
    end_sec: float,
    pipeline_cfg: PipelineConfig,
    pathroot: Path,
    features_pathroot: Path | None,
) -> tuple[np.ndarray, list[str]]:
    duration_s = float(end_sec) - float(start_sec)
    if duration_s <= 0.0:
        raise ValueError("Soundscape segment duration must be > 0")

    mel_segment = None
    if features_pathroot is not None:
        mel_candidate = features_pathroot / Path(filename).with_suffix(".npy")
        if mel_candidate.exists():
            mel_full = np.load(mel_candidate)
            mel_segment = _slice_mel_spectrogram_by_seconds(
                mel_full,
                start_sec,
                end_sec,
                pipeline_cfg.feature.sr,
                pipeline_cfg.feature.hop_length,
            )
            if mel_segment.shape[1] == 0:
                mel_segment = None

    if mel_segment is None:
        y = load_audio(
            pathroot / filename,
            sr=pipeline_cfg.feature.sr,
            offset=float(start_sec),
            duration=duration_s,
        )
        return build_feature_vector(y=y, pipeline_cfg=pipeline_cfg)

    return build_feature_vector(
        y=None,
        pipeline_cfg=pipeline_cfg,
        mel_spectrogram=mel_segment,
    )


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


def reduce_feature_memmap(run_path: Path, target_mel_bins: int = 32, soundscapes: bool = False) -> tuple[np.ndarray, list[str]]:
    """
    Reduce saved memmap artifacts.
    Saves reduced artifacts with `_reduced` suffix in same folder.
    """
    suffix = "_soundscape" if soundscapes else ""
    feature_names = np.load(run_path / f"feature_names{suffix}.npy", allow_pickle=True)
    X_shape = tuple(np.load(run_path / f"shape_X{suffix}.npy"))
    y_shape = tuple(np.load(run_path / f"shape_y{suffix}.npy"))
    X_dtype = np.load(run_path / f"dtype_X{suffix}.npy").item()
    y_dtype = np.load(run_path / f"dtype_y{suffix}.npy").item()

    X = np.memmap(run_path / f"X{suffix}.dat", dtype=X_dtype, mode="r", shape=(X_shape[0], X_shape[1]))
    y = np.memmap(run_path / f"y{suffix}.dat", dtype=y_dtype, mode="r", shape=y_shape)

    X_reduced, reduced_names = reduce_feature_vector(
        np.asarray(X, dtype=np.float32),
        feature_names,
        target_mel_bins=target_mel_bins,
    )

    X_out = np.memmap(
        run_path / f"X{suffix}_reduced.dat",
        mode="w+",
        dtype=np.float32,
        shape=X_reduced.shape,
    )
    X_out[:] = X_reduced
    X_out.flush()

    y_out = np.memmap(
        run_path / f"y{suffix}_reduced.dat",
        mode="w+",
        dtype=y_dtype,
        shape=y_shape,
    )
    y_out[:] = y
    y_out.flush()

    np.save(run_path / f"feature_names{suffix}_reduced.npy", np.asarray(reduced_names, dtype=object))
    np.save(run_path / f"file_ids{suffix}_reduced.npy", np.load(run_path / f"file_ids{suffix}.npy", allow_pickle=True))
    np.save(run_path / f"shape_X{suffix}_reduced.npy", np.array(X_reduced.shape))
    np.save(run_path / f"shape_y{suffix}_reduced.npy", np.array(y_shape))
    np.save(run_path / f"dtype_X{suffix}_reduced.npy", np.array(str(np.dtype(np.float32))))
    np.save(run_path / f"dtype_y{suffix}_reduced.npy", np.array(str(np.dtype(y_dtype))))

    return X_reduced, reduced_names


def _build_instance_pipeline_cfg(pipeline_cfg: PipelineConfig) -> PipelineConfig:
    if not pipeline_cfg.mil_mode:
        return pipeline_cfg
    return PipelineConfig(
        feature=pipeline_cfg.feature,
        pooling=pipeline_cfg.pooling,
        chunk=ChunkConfig(
            chunk_size_s=pipeline_cfg.mil.chunk_size_s,
            step_size_s=pipeline_cfg.mil.step_size_s,
        ),
        mil=pipeline_cfg.mil,
        mil_mode=pipeline_cfg.mil_mode,
    )


def _build_bags_meta_for_duration(
    duration_s: float,
    pipeline_cfg: PipelineConfig,
) -> tuple[np.ndarray, int]:
    bag_intervals = get_duration_chunk_intervals(
        duration_s,
        pipeline_cfg.chunk,
        pipeline_cfg.feature.sr,
        pipeline_cfg.feature.hop_length,
    )
    instance_intervals = get_duration_chunk_intervals(
        duration_s,
        pipeline_cfg.mil,
        pipeline_cfg.feature.sr,
        pipeline_cfg.feature.hop_length,
    )

    rows: list[tuple[int, int]] = []
    bag_id = 0
    chunk_id = 0
    for start, _ in instance_intervals:
        while bag_id + 1 < len(bag_intervals) and start >= bag_intervals[bag_id][1]:
            bag_id += 1
            chunk_id = 0
        rows.append((bag_id, chunk_id))
        chunk_id += 1

    n_used_bags = rows[-1][0] + 1 if rows else 0
    return np.asarray(rows, dtype=np.int32), n_used_bags


def build_feature_memmap_artifacts(
    df: pd.DataFrame,
    pipeline_cfg: PipelineConfig,
    pathroot: Path,
    filename_col: str,
    features_pathroot: Path,
    out_instances_path: Path,
    profiles_path: Path | None = None,
    species_ids_path: Path | None = None
):
    instance_pipeline_cfg = _build_instance_pipeline_cfg(pipeline_cfg)

    #  Infer dimensions
    sample_matrix, base_feature_names = extract_features_from_path(
        df[filename_col].iloc[0],
        pipeline_cfg=instance_pipeline_cfg,
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

    print("Features", len(feature_names))
    n_features = len(feature_names)
    bags_meta_rows: list[tuple[np.ndarray, int]] = []
    if pipeline_cfg.mil_mode:
        n_instances = 0
        for duration_s in df["duration"].values:
            row_bags_meta = _build_bags_meta_for_duration(duration_s, pipeline_cfg)
            bags_meta_rows.append(row_bags_meta)
            n_instances += row_bags_meta[0].shape[0]
    else:
        n_instances = count_nb_chunks(
            df,
            instance_pipeline_cfg.chunk,
            instance_pipeline_cfg.feature.sr,
            instance_pipeline_cfg.feature.hop_length,
        )

    print("Total number of instances", n_instances, "number of features", len(feature_names))
    out_instances_path.mkdir(parents=True, exist_ok=True)
    X = np.memmap(
        out_instances_path / "X.dat",
        mode="w+",
        dtype=np.float32,
        shape=(n_instances, n_features)
    )

    y = np.memmap(
        out_instances_path / "y.dat",
        mode="w+",
        dtype=np.int32,
        shape=(n_instances, 2)
    )

    # Fill memmaps
    file_ids = np.empty(n_instances, dtype=object)
    bags_meta = (
        np.empty((n_instances, 2), dtype=np.int32)
        if pipeline_cfg.mil_mode
        else None
    )

    cursor = 0
    bag_offset = 0
    for row_idx, row in enumerate(df.itertuples(index=False)):
        print(f"Cursor {cursor}/{n_instances}")
        filename = getattr(row, filename_col)

        matrix, _ = extract_features_from_path(
            getattr(row, filename_col),
            instance_pipeline_cfg,
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
        if pipeline_cfg.mil_mode and bags_meta is not None:
            row_bags_meta, n_bags = bags_meta_rows[row_idx]
            if n_chunks != row_bags_meta.shape[0]:
                raise RuntimeError(
                    "MIL chunk count mismatch between pooled features and bags metadata "
                    f"for {filename} ({n_chunks} != {row_bags_meta.shape[0]})."
                )
            row_bags_meta = row_bags_meta.copy()
            row_bags_meta[:, 0] += bag_offset
            bags_meta[cursor:cursor+n_chunks, :] = row_bags_meta
            bag_offset += n_bags

        X[cursor:cursor+n_chunks, :] = np.asarray(matrix, dtype=np.float32)
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
    if bags_meta is not None:
        np.save(out_instances_path / "bags_meta.npy", bags_meta)


def build_soundscape_feature_memmap_artifacts(
    df: pd.DataFrame,
    pipeline_cfg: PipelineConfig,
    pathroot: Path,
    features_pathroot: Path,
    out_instances_path: Path,
    filename_col: str = "filename",
    start_col: str = "start_sec",
    end_col: str = "end_sec",
    profiles_path: Path | None = None,
    species_ids_path: Path | None = None,
    n_classes: int = 234
):
    instance_pipeline_cfg = _build_instance_pipeline_cfg(pipeline_cfg)

    first_row = df.iloc[0]
    sample_matrix, base_feature_names = _extract_soundscape_segment_features(
        filename=str(first_row[filename_col]),
        start_sec=float(first_row[start_col]),
        end_sec=float(first_row[end_col]),
        pipeline_cfg=instance_pipeline_cfg,
        pathroot=pathroot,
        features_pathroot=features_pathroot,
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

    print("Features", len(feature_names))
    n_features = len(feature_names)
    bags_meta_rows: list[tuple[np.ndarray, int]] = []
    if pipeline_cfg.mil_mode:
        n_instances = 0
        for row in df.itertuples(index=False):
            start_sec = float(getattr(row, start_col))
            end_sec = float(getattr(row, end_col))
            duration_s = end_sec - start_sec
            row_bags_meta = _build_bags_meta_for_duration(duration_s, pipeline_cfg)
            bags_meta_rows.append(row_bags_meta)
            n_instances += row_bags_meta[0].shape[0]
    else:
        n_instances = 0
        for row in df.itertuples(index=False):
            start_sec = float(getattr(row, start_col))
            end_sec = float(getattr(row, end_col))
            duration_s = end_sec - start_sec
            intervals = get_duration_chunk_intervals(
                duration_s,
                instance_pipeline_cfg.chunk,
                instance_pipeline_cfg.feature.sr,
                instance_pipeline_cfg.feature.hop_length,
            )
            n_instances += len(intervals)

    print("Total number of instances", n_instances, "number of features", len(feature_names))
    out_instances_path.mkdir(parents=True, exist_ok=True)
    X = np.memmap(
        out_instances_path / "X_soundscape.dat",
        mode="w+",
        dtype=np.float32,
        shape=(n_instances, n_features)
    )

    y = np.memmap(
        out_instances_path / "y_soundscape.dat",
        mode="w+",
        dtype=np.int8,
        shape=(n_instances, 2, n_classes)  # [:, 0, :] primary label & [:, 1, :] class name
    )

    # Fill memmaps
    file_ids = np.empty(n_instances, dtype=object)
    bags_meta = (
        np.empty((n_instances, 2), dtype=np.int32)
        if pipeline_cfg.mil_mode
        else None
    )

    cursor = 0
    bag_offset = 0
    for row_idx, row in enumerate(df.itertuples(index=False)):
        print(f"Cursor {cursor}/{n_instances}")
        filename = getattr(row, filename_col)
        start_sec = float(getattr(row, start_col))
        end_sec = float(getattr(row, end_col))

        matrix, _ = _extract_soundscape_segment_features(
            filename=str(filename),
            start_sec=start_sec,
            end_sec=end_sec,
            pipeline_cfg=instance_pipeline_cfg,
            pathroot=pathroot,
            features_pathroot=features_pathroot,
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
        if pipeline_cfg.mil_mode and bags_meta is not None:
            row_bags_meta, n_bags = bags_meta_rows[row_idx]
            if n_chunks != row_bags_meta.shape[0]:
                raise RuntimeError(
                    "MIL chunk count mismatch between pooled features and bags metadata "
                    f"for {filename} ({n_chunks} != {row_bags_meta.shape[0]})."
                )
            row_bags_meta = row_bags_meta.copy()
            row_bags_meta[:, 0] += bag_offset
            bags_meta[cursor:cursor+n_chunks, :] = row_bags_meta
            bag_offset += n_bags

        X[cursor:cursor+n_chunks, :] = np.asarray(matrix, dtype=np.float32)
        pl_indices = [int(v) for v in getattr(row, "primary_label_int_list")]
        cn_indices = [int(v) for v in getattr(row, "class_name_int_list")]

        vec_pl = np.zeros(n_classes, dtype=np.int8)
        vec_cn = np.zeros(n_classes, dtype=np.int8)

        vec_pl[pl_indices] = 1
        vec_cn[cn_indices] = 1

        y[cursor:cursor+n_chunks, 0, :] = vec_cn
        y[cursor:cursor+n_chunks, 1, :] = vec_pl

        file_ids[cursor:cursor+n_chunks] = filename
        cursor += n_chunks

    X.flush()
    y.flush()

    # Metadata
    np.save(out_instances_path / "feature_names_soundscape.npy", feature_names)
    np.save(out_instances_path / "file_ids_soundscape.npy", file_ids)
    np.save(out_instances_path / "shape_X_soundscape.npy", np.array(X.shape))
    np.save(out_instances_path / "shape_y_soundscape.npy", np.array(y.shape))
    np.save(out_instances_path / "dtype_X_soundscape.npy", np.array(str(X.dtype)))
    np.save(out_instances_path / "dtype_y_soundscape.npy", np.array(str(y.dtype)))
    if bags_meta is not None:
        np.save(out_instances_path / "bags_meta_soundscape.npy", bags_meta)
