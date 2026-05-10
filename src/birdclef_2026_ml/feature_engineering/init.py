from pathlib import Path
import gc
import numpy as np

from birdclef_2026_ml.processing.audio_utils import build_audio_path, load_audio
from birdclef_2026_ml.configs import PipelineConfig, ChunkConfig, save_config
from birdclef_2026_ml.feature_engineering.utils import (
    pad_short_audio,
    _percentile_label,
    # joblib_to_frame_features,
)
from birdclef_2026_ml.feature_engineering.feature_extract import (
    compute_profile_similarity_features,
    compute_mel_spectrogram,
    extract_all_frame_features,
    extract_all_frame_features_from_mel,
)
from birdclef_2026_ml.feature_engineering.pooling import pool_feature_dict, pool_feature_dict_sliding_windows
from birdclef_2026_ml.feature_engineering.chunking import get_sliding_window_intervals


def extract_save_mel_spectograms(
    df,
    input_path: Path,
    pipeline_cfg: PipelineConfig,
    config_path: Path,
    output_path: Path,
) -> int:
    """Compute mel spectrogram only, save raw non-aggregated arrays as .npy."""

    pipeline_cfg.feature = pipeline_cfg.feature
    output_path.mkdir(parents=True, exist_ok=True)

    filenames = df["filename"].unique()
    total = len(filenames)

    save_config(pipeline_cfg.feature, config_path)

    for i, filename in enumerate(filenames, start=1):
        y = load_audio(input_path / filename, sr=pipeline_cfg.feature.sr)
        mel_spectrogram_db = compute_mel_spectrogram(y, pipeline_cfg.feature)

        saved_count = 0
        out_path = (output_path / filename).with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, mel_spectrogram_db)
        saved_count += 1

        print(f"[{i}/{total}] saved {saved_count} feature arrays for {filename}")

        del y
        del mel_spectrogram_db
        gc.collect()

    return total


def add_profile_similarity_features(
    X: np.ndarray,
    feature_names: list[str],
    profiles_path: Path,
    species_ids_path: Path,
) -> tuple[np.ndarray, list[str]]:
    """
    Append cosine-similarity-to-profile features to pooled feature matrix.
    """
    species_profiles = np.load(profiles_path)
    species_ids = np.load(species_ids_path)

    sim_matrix, sim_feature_names = compute_profile_similarity_features(
        X=np.asarray(X, dtype=float),
        feature_names=feature_names,
        species_profiles=species_profiles,
        species_ids=species_ids,
    )
    X_aug = np.hstack([np.asarray(X, dtype=float), sim_matrix])
    return X_aug, [*feature_names, *sim_feature_names]


def _resolve_frame_features(
    *,
    y: np.ndarray | None,
    pipeline_cfg: PipelineConfig,
    frame_features: dict[str, np.ndarray] | None = None,
    # frame_features_joblib_path: str | Path | None = None,
    mel_spectrogram: np.ndarray | None = None,
    mel_spectrogram_npy_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    provided = sum(
        value is not None
        for value in (frame_features, mel_spectrogram, mel_spectrogram_npy_path, y)
    )
    if provided != 1:
        raise ValueError("Provide exactly one feature source among y/frame_features/joblib/mel/mel_path.")

    # if frame_features is not None:
    #     return joblib_to_frame_features(frame_features)

    # if frame_features_joblib_path is not None:
    #     return joblib_to_frame_features(frame_features_joblib_path)

    if mel_spectrogram is not None:
        return extract_all_frame_features_from_mel(mel_spectrogram, pipeline_cfg.feature)

    if mel_spectrogram_npy_path is not None:
        mel_spectrogram_db = np.load(mel_spectrogram_npy_path)
        return extract_all_frame_features_from_mel(mel_spectrogram_db, pipeline_cfg.feature)

    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.size == 0:
        raise ValueError("y must contain at least one sample")
    return extract_all_frame_features(y_arr, pipeline_cfg.feature)


def get_feature_names(
    features_dict: dict[str, np.ndarray],
    pipeline_cfg: PipelineConfig,
) -> list[str]:
    """Return feature names in the exact same order as pooled outputs."""
    names: list[str] = []

    for feature_name, feature_matrix in features_dict.items():
        x = np.asarray(feature_matrix)
        n_dims = 1 if x.ndim == 1 else int(x.shape[0])

        base_names: list[str] = []
        for stat in pipeline_cfg.pooling.stats:
            for dim_idx in range(1, n_dims + 1):
                base_names.append(f"{feature_name}_{dim_idx:02d}_{stat}")

        for dim_idx in range(1, n_dims + 1):
            for percentile in pipeline_cfg.pooling.percentiles:
                p_label = _percentile_label(percentile)
                base_names.append(f"{feature_name}_{dim_idx:02d}_p{p_label}")

        if pipeline_cfg.pooling.pooling_mode == "global":
            names.extend(base_names)

        if pipeline_cfg.pooling.pooling_mode == "chunk":
            names.extend(base_names)

    return names


def build_feature_vector(
    y: np.ndarray | None,
    pipeline_cfg: PipelineConfig,
    frame_features: dict[str, np.ndarray] | None = None,
    # frame_features_joblib_path: Path | None = None,
    mel_spectrogram: np.ndarray | None = None,
    mel_spectrogram_npy_path: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build pooled features and aligned feature names.

    - global mode: returns 1D vector
    - chunk mode: returns 2D matrix (n_chunks, n_features)
    """
    frame_features_resolved = _resolve_frame_features(
        y=y,
        pipeline_cfg=pipeline_cfg,
        frame_features=frame_features,
        # frame_features_joblib_path=frame_features_joblib_path,
        mel_spectrogram=mel_spectrogram,
        mel_spectrogram_npy_path=mel_spectrogram_npy_path,
    )

    pooled = pool_feature_dict(
        features_dict=frame_features_resolved,
        pipeline_cfg=pipeline_cfg
    )

    feature_names = get_feature_names(
        features_dict=frame_features_resolved,
        pipeline_cfg=pipeline_cfg
    )

    if pipeline_cfg.pooling.pooling_mode == "global":
        vectors: list[np.ndarray] = [np.asarray(v, dtype=float).ravel() for v in pooled.values()]
        if vectors:
            feature_vector = np.concatenate(vectors)
        else:
            feature_vector = np.array([], dtype=float)

        if feature_vector.size != len(feature_names):
            raise RuntimeError(
                "Feature vector size and feature name count mismatch "
                f"({feature_vector.size} != {len(feature_names)})."
            )
        return feature_vector, feature_names

    chunk_mats: list[np.ndarray] = []
    # n_chunks = None
    for pooled_mat in pooled.values():
        mat = np.asarray(pooled_mat, dtype=float)
        # if mat.ndim == 1:
        #     mat = mat[np.newaxis, :]
        # if mat.ndim != 2:
        #     raise RuntimeError("Chunk pooled feature must be 2D")
        chunk_mats.append(mat)

    if not chunk_mats:
        return np.empty((0, 0), dtype=float), feature_names

    # Concatenate all features for each chunk
    feature_matrix = np.hstack(chunk_mats) if chunk_mats else np.empty((0, len(feature_names)), dtype=float)
    if feature_matrix.shape[1] != len(feature_names):
        raise RuntimeError(
            "Chunk feature matrix width and feature name count mismatch "
            f"({feature_matrix.shape[1]} != {len(feature_names)})."
        )

    return feature_matrix, feature_names


def split_audio_into_chunks(
    y: np.ndarray,
    pipeline_cfg: PipelineConfig,
) -> list[np.ndarray]:
    """Split audio into fixed-duration chunks using the shared chunking logic."""
    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.size == 0:
        return []

    pipeline_cfg.feature = pipeline_cfg.feature
    pipeline_cfg.chunk = pipeline_cfg.chunk

    y_arr = pad_short_audio(y_arr, pipeline_cfg.chunk.chunk_size_s, pipeline_cfg.feature.sr)
    chunk_intervals = get_sliding_window_intervals(
        n_frames=y_arr.shape[0],
        window_size_s=pipeline_cfg.chunk.chunk_size_s,
        step_size_s=pipeline_cfg.chunk.step_size_s,
        frame_rate_hz=float(pipeline_cfg.feature.sr),
    )
    return [y_arr[start:end] for start, end in chunk_intervals if end > start]


def build_mil_feature_matrix(
    y: np.ndarray | None,
    pipeline_cfg: PipelineConfig,
    # pipeline_cfg.feature: FeatureConfig,
    # pipeline_cfg.pooling: PoolingConfig,
    # mil_cfg: MILConfig,
    frame_features: dict[str, np.ndarray] | None = None,
    frame_features_joblib_path: str | Path | None = None,
    mel_spectrogram: np.ndarray | None = None,
    mel_spectrogram_npy_path: str | Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build one MIL bag from one chunk of audio."""
    pipeline_cfg.feature = pipeline_cfg.feature
    pipeline_cfg.pooling = pipeline_cfg.pooling
    mil_cfg = pipeline_cfg.mil

    frame_features_resolved = _resolve_frame_features(
        y=y,
        pipeline_cfg=pipeline_cfg,
        frame_features=frame_features,
        # frame_features_joblib_path=frame_features_joblib_path,
        mel_spectrogram=mel_spectrogram,
        mel_spectrogram_npy_path=mel_spectrogram_npy_path,
    )

    pooled = pool_feature_dict_sliding_windows(
        features_dict=frame_features_resolved,
        pipeline_cfg=pipeline_cfg,
        # window_size_s=mil_cfg.window_size_s,
        # step_size_s=mil_cfg.step_size_s,
    )
    feature_names = get_feature_names(
        features_dict=frame_features_resolved,
        pipeline_cfg=PipelineConfig(
            feature=pipeline_cfg.feature,
            pooling=pipeline_cfg.pooling,
            chunk=ChunkConfig(chunk_size_s=mil_cfg.window_size_s, step_size_s=mil_cfg.window_size_s),
            mil=mil_cfg
        )
    )

    window_mats: list[np.ndarray] = []
    n_windows = 0
    for pooled_mat in pooled.values():
        mat = np.asarray(pooled_mat, dtype=float)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        if mat.ndim != 2:
            raise RuntimeError("MIL pooled feature must be 2D")
        window_mats.append(mat)
        n_windows = max(n_windows, mat.shape[0])

    if not window_mats:
        return np.empty((0, 0), dtype=float), feature_names

    fill_value = float(pipeline_cfg.pooling.nan_fill_value)
    rows: list[np.ndarray] = []
    for window_idx in range(n_windows):
        row_parts: list[np.ndarray] = []
        for mat in window_mats:
            if window_idx < mat.shape[0]:
                row_parts.append(mat[window_idx])
            else:
                row_parts.append(np.full(mat.shape[1], fill_value, dtype=float))
        rows.append(np.concatenate(row_parts) if row_parts else np.array([], dtype=float))

    feature_matrix = np.vstack(rows) if rows else np.empty((0, len(feature_names)), dtype=float)
    if feature_matrix.shape[1] != len(feature_names):
        raise RuntimeError(
            "MIL feature matrix width and feature name count mismatch "
            f"({feature_matrix.shape[1]} != {len(feature_names)})."
        )
    return feature_matrix, feature_names


def extract_features_from_path(
    audio_path: str,
    pipeline_cfg: PipelineConfig,
    pathroot: Path,
    features_pathroot: Path | None,
) -> tuple[np.ndarray, list[str]]:
    """Load one audio file and return pooled features + names."""

    y = None
    mel_npy_path = None
    if features_pathroot is not None:
        mel_candidate = features_pathroot / Path(audio_path).with_suffix(".npy")
        if mel_candidate.exists():
            mel_npy_path = mel_candidate
    else:
        y = load_audio(pathroot / audio_path, sr=pipeline_cfg.feature.sr)

    return build_feature_vector(
        y=y,
        pipeline_cfg=pipeline_cfg,
        # frame_features_joblib_path=features_joblib_path,
        mel_spectrogram_npy_path=mel_npy_path,
    )


def extract_features_from_row(
    df,
    idx: int,
    *,
    pipeline_cfg: PipelineConfig,
    pathroot: Path,
    filename_col: str = "filename",
    features_pathroot: Path | None,
) -> tuple[np.ndarray, list[str]]:
    """Load one audio file and return pooled features + names."""
    y = None
    mel_npy_path = None
    if features_pathroot is not None:
        filename = df[filename_col].iloc[idx]
        mel_candidate = features_pathroot / Path(filename).with_suffix(".npy")
        if mel_candidate.exists():
            mel_npy_path = mel_candidate
    else:
        path = build_audio_path(
            df=df,
            idx=idx,
            pathroot=pathroot,
            filename_col=filename_col,
        )
        y = load_audio(path, sr=pipeline_cfg.feature.sr)
    return build_feature_vector(
        y=y,
        pipeline_cfg=pipeline_cfg,
        # frame_features_joblib_path=features_joblib_path,
        mel_spectrogram_npy_path=mel_npy_path,
    )


def build_feature_matrix_from_df(
    df,
    *,
    pipeline_cfg: PipelineConfig,
    indices: list[int] | None = None,
    pathroot: Path,
    filename_col: str = "filename",
    features_pathroot: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a 2D feature matrix from dataframe rows.

    - global mode: one output row per audio row
    - chunk mode: one output row per chunk (audio rows are expanded)
    """
    if indices is None:
        indices = list(range(len(df)))

    if len(indices) == 0:
        return np.empty((0, 0), dtype=float), []

    vectors: list[np.ndarray] = []
    feature_names_ref: list[str] | None = None

    for idx in indices:
        vec, names = extract_features_from_row(
            df=df,
            idx=idx,
            pipeline_cfg=pipeline_cfg,
            pathroot=pathroot,
            filename_col=filename_col,
            features_pathroot=features_pathroot,
        )

        if pipeline_cfg.pooling.pooling_mode == "global":
            if feature_names_ref is None:
                feature_names_ref = names
            elif names != feature_names_ref:
                raise RuntimeError(
                    "Inconsistent feature names across rows. "
                    "Ensure all rows produce vectors with the same schema."
                )

            vectors.append(np.asarray(vec, dtype=float).ravel())
            continue

        mat = np.asarray(vec, dtype=float)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        if mat.ndim != 2:
            raise RuntimeError("Chunk features must be a 2D matrix")

        if mat.shape[1] != len(names):
            raise RuntimeError(
                "Chunk feature width and names length mismatch "
                f"({mat.shape[1]} != {len(names)})."
            )

        if feature_names_ref is None:
            feature_names_ref = list(names)
        elif names != feature_names_ref:
            raise RuntimeError(
                "Inconsistent feature names across rows in chunk mode. "
                "Ensure chunk features are produced with the same schema."
            )

        for row in mat:
            vectors.append(np.asarray(row, dtype=float).ravel())

    if not vectors:
        return np.empty((0, 0), dtype=float), []

    feature_matrix = np.vstack(vectors)
    return feature_matrix, (feature_names_ref or [])


def build_feature_matrix_and_labels_from_df(
    df,
    *,
    label_cols: str | list[str],
    pipeline_cfg: PipelineConfig,
    indices: list[int] | None = None,
    pathroot: Path,
    filename_col: str = "filename",
    features_pathroot: Path | None = None,
) -> tuple[np.ndarray, np.ndarray | dict[str, np.ndarray], list[str]]:
    """Build feature matrix and aligned labels.

    In chunk mode, each audio row can produce multiple feature rows.
    Labels are repeated to match the expanded number of rows.
    """
    if isinstance(label_cols, str):
        label_names = [label_cols]
        return_single = True
    else:
        label_names = list(label_cols)
        return_single = False

    if not label_names:
        raise ValueError("label_cols must contain at least one column name")

    missing_cols = [col for col in label_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing label columns in df: {missing_cols}")

    if indices is None:
        indices = list(range(len(df)))

    if len(indices) == 0:
        empty_labels = {col: np.array([], dtype=object) for col in label_names}
        if return_single:
            return np.empty((0, 0), dtype=float), empty_labels[label_names[0]], []
        return np.empty((0, 0), dtype=float), empty_labels, []

    vectors: list[np.ndarray] = []
    feature_names_ref: list[str] | None = None
    labels_expanded: dict[str, list[object]] = {col: [] for col in label_names}

    for idx in indices:
        vec, names = extract_features_from_row(
            df=df,
            idx=idx,
            pipeline_cfg=pipeline_cfg,
            pathroot=pathroot,
            filename_col=filename_col,
            features_pathroot=features_pathroot,
        )

        if pipeline_cfg.pooling.pooling_mode == "global":
            row_matrix = np.asarray(vec, dtype=float).ravel()[np.newaxis, :]
        if pipeline_cfg.pooling.pooling_mode == "chunk":
            row_matrix = np.asarray(vec, dtype=float)
            if row_matrix.ndim == 1:
                row_matrix = row_matrix[np.newaxis, :]
            if row_matrix.ndim != 2:
                raise RuntimeError("Chunk features must be a 2D matrix")

        if row_matrix.shape[1] != len(names):
            raise RuntimeError(
                "Feature width and names length mismatch "
                f"({row_matrix.shape[1]} != {len(names)})."
            )

        if feature_names_ref is None:
            feature_names_ref = list(names)
        elif names != feature_names_ref:
            raise RuntimeError(
                "Inconsistent feature names across rows. "
                "Ensure all rows produce features with the same schema."
            )

        for row in row_matrix:
            vectors.append(np.asarray(row, dtype=float).ravel())

        n_repeats = row_matrix.shape[0]
        row_labels = df.iloc[idx]
        for col in label_names:
            labels_expanded[col].extend([row_labels[col]] * n_repeats)

    if not vectors:
        empty_labels = {col: np.array([], dtype=object) for col in label_names}
        if return_single:
            return np.empty((0, 0), dtype=float), empty_labels[label_names[0]], []
        return np.empty((0, 0), dtype=float), empty_labels, []

    feature_matrix = np.vstack(vectors)
    labels_out = {col: np.asarray(vals) for col, vals in labels_expanded.items()}

    for col, arr in labels_out.items():
        if arr.shape[0] != feature_matrix.shape[0]:
            raise RuntimeError(
                "Expanded labels and feature matrix row count mismatch "
                f"for '{col}' ({arr.shape[0]} != {feature_matrix.shape[0]})."
            )

    if return_single:
        return feature_matrix, labels_out[label_names[0]], (feature_names_ref or [])
    return feature_matrix, labels_out, (feature_names_ref or [])
