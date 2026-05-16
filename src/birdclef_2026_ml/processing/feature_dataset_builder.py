import json
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
from birdclef_2026_ml.processing.memmap_dataset import load_memmap_dataset


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


def _write_memmap_metadata(
    metadata_path: Path,
    *,
    feature_names: list[str],
    file_ids: np.ndarray | list[str],
    X_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    X_dtype: np.dtype,
    y_dtype: np.dtype,
    X_path: str,
    y_path: str,
    bags_meta: np.ndarray | None = None,
    bags_meta_path: str | None = None,
) -> None:
    metadata = {
        "X": {
            "path": X_path,
            "shape": [int(value) for value in X_shape],
            "dtype": np.dtype(X_dtype).str,
        },
        "y": {
            "path": y_path,
            "shape": [int(value) for value in y_shape],
            "dtype": np.dtype(y_dtype).str,
        },
        "feature_names": [str(name) for name in feature_names],
        "file_ids": [str(value) for value in file_ids],
    }

    if bags_meta is not None and bags_meta_path is not None:
        metadata["bags_meta"] = {
            "path": bags_meta_path,
            "shape": [int(value) for value in bags_meta.shape],
            "dtype": np.dtype(bags_meta.dtype).str,
        }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _build_mel_reduction_step(
    feature_names: list[str],
    prefix: str,
    target_bins: int,
) -> tuple[list[int], list[str], list[list[int]], list[str]]:
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

    group_cols: list[list[int]] = []
    group_names: list[str] = []
    for key in sorted(grouped.keys()):
        family, stat = key.split("|", 1)
        entries = sorted(grouped[key], key=lambda item: item[0])
        dim_groups = np.array_split(np.arange(len(entries)), target_bins)
        for out_dim, group in enumerate(dim_groups, start=1):
            if len(group) == 0:
                continue
            cols = [entries[i][1] for i in group]
            group_cols.append(cols)
            group_names.append(f"{family}_{out_dim:02d}_{stat}")

    return keep_indices, keep_names, group_cols, group_names


def _apply_mel_reduction_step(
    X: np.ndarray,
    keep_indices: list[int],
    group_cols: list[list[int]],
) -> np.ndarray:
    reduced_cols: list[np.ndarray] = []
    if keep_indices:
        reduced_cols.append(X[:, keep_indices])

    for cols in group_cols:
        reduced_cols.append(np.median(X[:, cols], axis=1, keepdims=True))

    if reduced_cols:
        return np.hstack(reduced_cols)
    return np.empty((X.shape[0], 0), dtype=float)


def _reduce_mel_family_features(
    X: np.ndarray,
    feature_names: list[str],
    prefix: str,
    target_bins: int = 32,
) -> tuple[np.ndarray, list[str]]:
    print(f"[reduce_feature_vector] reduce family='{prefix}' target_bins={target_bins}")
    keep_indices, keep_names, group_cols, group_names = _build_mel_reduction_step(
        feature_names,
        prefix=prefix,
        target_bins=target_bins,
    )
    reduced_names = keep_names + group_names
    X_reduced = _apply_mel_reduction_step(X, keep_indices, group_cols)
    print(
        f"[reduce_feature_vector] family done='{prefix}' keep={len(keep_indices)} "
        f"reduced_groups={len(group_cols)} out_cols={len(reduced_names)}"
    )
    return X_reduced, reduced_names


def reduce_feature_vector(
    X: np.ndarray,
    feature_names: list[str] | np.ndarray,
    target_mel_bins: int = 32,
    batch_rows: int | None = None,
    out: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Reduce feature matrix:
    - drop *_std for mel spectrogram families only
    - collapse mel/mel_delta/mel_delta2 dims from 128 -> target_mel_bins with median
    """
    print(
        f"[reduce_feature_vector] start rows={X.shape[0]} cols={X.shape[1]} "
        f"target_mel_bins={target_mel_bins}"
    )
    feature_names_list = [str(name) for name in feature_names]
    mel_std_prefixes = (
        "mel_spectrogram_",
        "mel_spectrogram_delta_",
        "mel_spectrogram_delta2_",
    )
    keep_mask = np.array(
        [not (name.endswith("_std") and name.startswith(mel_std_prefixes)) for name in feature_names_list],
        dtype=bool,
    )
    reduced_names = [name for name, keep in zip(feature_names_list, keep_mask) if keep]
    dropped = int(np.size(keep_mask) - int(np.sum(keep_mask)))
    print(
        f"[reduce_feature_vector] drop mel *_std removed={dropped} kept={len(reduced_names)}"
    )

    if batch_rows is None and isinstance(X, np.memmap):
        keep_count = max(1, int(np.sum(keep_mask)))
        target_bytes = 128 * 1024 * 1024
        batch_rows = max(1, min(X.shape[0], int(target_bytes / (keep_count * 4))))

    if batch_rows is None or X.shape[0] == 0:
        X_reduced = np.asarray(X[:, keep_mask], dtype=np.float32)
        for prefix in ("mel_spectrogram_", "mel_spectrogram_delta_", "mel_spectrogram_delta2_"):
            X_reduced, reduced_names = _reduce_mel_family_features(
                X_reduced,
                reduced_names,
                prefix=prefix,
                target_bins=target_mel_bins,
            )

        print(
            f"[reduce_feature_vector] done rows={X_reduced.shape[0]} cols={X_reduced.shape[1]}"
        )
        return X_reduced, reduced_names

    keep_indices = np.flatnonzero(keep_mask).tolist()
    steps: list[dict[str, object]] = []
    for prefix in ("mel_spectrogram_", "mel_spectrogram_delta_", "mel_spectrogram_delta2_"):
        keep_idx, keep_names, group_cols, group_names = _build_mel_reduction_step(
            reduced_names,
            prefix=prefix,
            target_bins=target_mel_bins,
        )
        steps.append(
            {
                "prefix": prefix,
                "keep_indices": keep_idx,
                "group_cols": group_cols,
                "keep_count": len(keep_idx),
                "group_count": len(group_cols),
                "out_cols": len(keep_names) + len(group_names),
            }
        )
        reduced_names = keep_names + group_names

    if out is None:
        X_out = np.empty((X.shape[0], len(reduced_names)), dtype=np.float32)
    else:
        X_out = out
        if X_out.shape != (X.shape[0], len(reduced_names)):
            raise ValueError("Output shape does not match reduced feature shape.")

    for step in steps:
        print(
            "[reduce_feature_vector] reduce family='{prefix}' target_bins={bins}".format(
                prefix=step["prefix"],
                bins=target_mel_bins,
            )
        )
        print(
            "[reduce_feature_vector] family plan='{prefix}' keep={keep} reduced_groups={groups} "
            "out_cols={out_cols}".format(
                prefix=step["prefix"],
                keep=step["keep_count"],
                groups=step["group_count"],
                out_cols=step["out_cols"],
            )
        )

    for start in range(0, X.shape[0], batch_rows):
        end = min(start + batch_rows, X.shape[0])
        X_batch = np.asarray(X[start:end, :][:, keep_indices], dtype=np.float32)
        for step in steps:
            X_batch = _apply_mel_reduction_step(
                X_batch,
                step["keep_indices"],
                step["group_cols"],
            )
        X_out[start:end, :] = X_batch

    print(
        f"[reduce_feature_vector] done rows={X_out.shape[0]} cols={X_out.shape[1]}"
    )
    return X_out, reduced_names


def reduce_feature_memmap(run_path: Path, target_mel_bins: int = 32, soundscapes: bool = False) -> tuple[np.ndarray, list[str]]:
    """
    Reduce saved memmap artifacts.
    """
    suffix = "_soundscape" if soundscapes else ""
    run_name = run_path.parent.name
    dataset = load_memmap_dataset(run_name=run_name, reduced=False, soundscape=soundscapes)
    feature_names = dataset.feature_names
    X = dataset.X
    y = dataset.y
    X_shape = X.shape
    y_shape = y.shape

    feature_names_list = [str(name) for name in feature_names]
    mel_std_prefixes = (
        "mel_spectrogram_",
        "mel_spectrogram_delta_",
        "mel_spectrogram_delta2_",
    )
    keep_mask = np.array(
        [not (name.endswith("_std") and name.startswith(mel_std_prefixes)) for name in feature_names_list],
        dtype=bool,
    )
    reduced_names = [name for name, keep in zip(feature_names_list, keep_mask) if keep]
    for prefix in ("mel_spectrogram_", "mel_spectrogram_delta_", "mel_spectrogram_delta2_"):
        _, keep_names, _, group_names = _build_mel_reduction_step(
            reduced_names,
            prefix=prefix,
            target_bins=target_mel_bins,
        )
        reduced_names = keep_names + group_names

    out_path = run_path / "reduced"
    X_out = np.memmap(
        out_path / f"X{suffix}.dat",
        mode="w+",
        dtype=np.float32,
        shape=(X_shape[0], len(reduced_names)),
    )

    X_reduced, reduced_names = reduce_feature_vector(
        X,
        feature_names,
        target_mel_bins=target_mel_bins,
        out=X_out,
    )
    if hasattr(X_reduced, "flush"):
        X_reduced.flush()

    y_out = np.memmap(
        out_path / f"y{suffix}.dat",
        mode="w+",
        dtype=y.dtype,
        shape=y_shape,
    )
    y_out[:] = y
    y_out.flush()

    bags_meta = dataset.bags_meta
    if bags_meta is not None:
        np.save(out_path / f"bags_meta{suffix}.npy", bags_meta)

    file_ids = dataset.filenames[:X_reduced.shape[0]]
    metadata_path = out_path / f"metadata{suffix}.json"
    _write_memmap_metadata(
        metadata_path,
        feature_names=reduced_names,
        file_ids=file_ids,
        X_shape=X_reduced.shape,
        y_shape=y_shape,
        X_dtype=np.float32,
        y_dtype=y.dtype,
        X_path=f"X{suffix}.dat",
        y_path=f"y{suffix}.dat",
        bags_meta=bags_meta,
        bags_meta_path=f"bags_meta{suffix}.npy" if bags_meta is not None else None,
    )

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
        pipeline_cfg.feature.n_fft,
        pipeline_cfg.feature.hop_length,
        with_pad=False,
    )
    instance_intervals = get_duration_chunk_intervals(
        duration_s,
        pipeline_cfg.mil,
        pipeline_cfg.feature.sr,
        pipeline_cfg.feature.n_fft,
        pipeline_cfg.feature.hop_length,
        with_pad=True
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
    n_instances = count_nb_chunks(
        df,
        instance_pipeline_cfg.chunk,
        instance_pipeline_cfg.feature.n_fft,
        instance_pipeline_cfg.feature.sr,
        instance_pipeline_cfg.feature.hop_length,
        with_pad=True
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
        np.empty(n_instances, dtype=np.int32)
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
            bags_meta[cursor:cursor + n_chunks] = np.arange(0, n_chunks)

        X[cursor:cursor+n_chunks, :] = np.asarray(matrix, dtype=np.float32)
        pl = getattr(row, "primary_label_int")
        cn = getattr(row, "class_name_int")

        y[cursor:cursor+n_chunks, 0] = cn
        y[cursor:cursor+n_chunks, 1] = pl

        file_ids[cursor:cursor+n_chunks] = filename
        cursor += n_chunks

    X.flush()
    y.flush()
    print(bags_meta, cursor)
    # Metadata
    X_shape = (cursor,) + X.shape[1:]
    y_shape = (cursor,) + y.shape[1:]
    file_ids = file_ids[:cursor]
    if bags_meta is not None:
        bags_meta = bags_meta[:cursor]
        np.save(out_instances_path / "bags_meta.npy", bags_meta)

    _write_memmap_metadata(
        out_instances_path / "metadata.json",
        feature_names=feature_names,
        file_ids=file_ids,
        X_shape=X_shape,
        y_shape=y_shape,
        X_dtype=X.dtype,
        y_dtype=y.dtype,
        X_path="X.dat",
        y_path="y.dat",
        bags_meta=bags_meta,
        bags_meta_path="bags_meta.npy" if bags_meta is not None else None,
    )


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
                instance_pipeline_cfg.feature.n_fft,
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
    X_shape = (cursor,) + X.shape[1:]
    y_shape = (cursor,) + y.shape[1:]
    file_ids = file_ids[:cursor]
    if bags_meta is not None:
        bags_meta = bags_meta[:cursor]
        np.save(out_instances_path / "bags_meta_soundscape.npy", bags_meta)

    _write_memmap_metadata(
        out_instances_path / "metadata_soundscape.json",
        feature_names=feature_names,
        file_ids=file_ids,
        X_shape=X_shape,
        y_shape=y_shape,
        X_dtype=X.dtype,
        y_dtype=y.dtype,
        X_path="X_soundscape.dat",
        y_path="y_soundscape.dat",
        bags_meta=bags_meta,
        bags_meta_path="bags_meta_soundscape.npy" if bags_meta is not None else None,
    )
