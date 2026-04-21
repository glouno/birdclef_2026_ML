from typing import Literal
import librosa
import numpy as np

from birdclef_2026_ml.audio_utils import build_audio_path, get_path, load_audio
from birdclef_2026_ml.feature_engineering.configs import FeatureConfig, PoolingConfig, ChunkConfig
from birdclef_2026_ml.feature_engineering.utils import _pad_short_audio_randomly, _percentile_label
from birdclef_2026_ml.feature_engineering.feature_extract import extract_all_frame_features
from birdclef_2026_ml.feature_engineering.pooling import pool_feature_dict


def get_feature_names(
    features_dict: dict[str, np.ndarray],
    pooling_cfg: PoolingConfig,
    pooling_mode: Literal["global", "chunk"],
    feature_cfg: FeatureConfig,
    chunk_cfg: ChunkConfig | None = None,
) -> list[str]:
    """Return feature names in the exact same order as pooled outputs."""
    names: list[str] = []

    for feature_name, feature_matrix in features_dict.items():
        x = np.asarray(feature_matrix)
        n_dims = 1 if x.ndim == 1 else int(x.shape[0])

        base_names: list[str] = []
        for stat in pooling_cfg.stats:
            for dim_idx in range(1, n_dims + 1):
                base_names.append(f"{feature_name}_{dim_idx:02d}_{stat}")

        for dim_idx in range(1, n_dims + 1):
            for percentile in pooling_cfg.percentiles:
                p_label = _percentile_label(percentile)
                base_names.append(f"{feature_name}_{dim_idx:02d}_p{p_label}")

        if pooling_mode == "global":
            names.extend(base_names)
        elif pooling_mode == "chunk":
            if chunk_cfg is None:
                raise ValueError("chunk_cfg is required for chunk mode")
            names.extend(base_names)
        else:
            raise ValueError(f"Unsupported pooling mode: {pooling_mode}")

    return names


def build_feature_vector(
    y: np.ndarray,
    feature_cfg: FeatureConfig,
    pooling_mode: Literal["global", "chunk"],
    pooling_cfg: PoolingConfig,
    chunk_cfg: ChunkConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build pooled features and aligned feature names.

    - global mode: returns 1D vector
    - chunk mode: returns 2D matrix (n_chunks, n_features)
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.size == 0:
        raise ValueError("y must contain at least one sample")

    if pooling_mode == "chunk":
        if chunk_cfg is None:
            raise ValueError("chunk_cfg is required for chunk mode")
        min_samples = int(np.ceil(float(chunk_cfg.chunks_s) * float(feature_cfg.sr)))
        y_arr = _pad_short_audio_randomly(y_arr, min_samples=min_samples)

    frame_features = extract_all_frame_features(y_arr, feature_cfg)
    pooled = pool_feature_dict(
        features_dict=frame_features,
        pooling_cfg=pooling_cfg,
        mode=pooling_mode,
        feature_cfg=feature_cfg,
        chunk_cfg=chunk_cfg,
    )

    feature_names = get_feature_names(
        features_dict=frame_features,
        pooling_cfg=pooling_cfg,
        pooling_mode=pooling_mode,
        feature_cfg=feature_cfg,
        chunk_cfg=chunk_cfg,
    )

    if pooling_mode == "global":
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

    if pooling_mode != "chunk":
        raise ValueError(f"Unsupported pooling mode: {pooling_mode}")

    chunk_mats: list[np.ndarray] = []
    n_chunks = 0
    for pooled_mat in pooled.values():
        mat = np.asarray(pooled_mat, dtype=float)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        if mat.ndim != 2:
            raise RuntimeError("Chunk pooled feature must be 2D")
        chunk_mats.append(mat)
        n_chunks = max(n_chunks, mat.shape[0])

    if not chunk_mats:
        return np.empty((0, 0), dtype=float), feature_names

    fill_value = float(pooling_cfg.nan_fill_value)
    rows: list[np.ndarray] = []
    for chunk_idx in range(n_chunks):
        row_parts: list[np.ndarray] = []
        for mat in chunk_mats:
            if chunk_idx < mat.shape[0]:
                row_parts.append(mat[chunk_idx])
            else:
                row_parts.append(np.full(mat.shape[1], fill_value, dtype=float))
        rows.append(np.concatenate(row_parts) if row_parts else np.array([], dtype=float))

    feature_matrix = np.vstack(rows) if rows else np.empty((0, len(feature_names)), dtype=float)
    if feature_matrix.shape[1] != len(feature_names):
        raise RuntimeError(
            "Chunk feature matrix width and feature name count mismatch "
            f"({feature_matrix.shape[1]} != {len(feature_names)})."
        )

    return feature_matrix, feature_names


def extract_features_from_path(
    audio_path: str,
    feature_cfg: FeatureConfig,
    pooling_mode: Literal["global", "chunk"],
    pooling_cfg: PoolingConfig,
    chunk_cfg: ChunkConfig | None = None,
    pathroot: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load one audio file and return pooled features + names."""
    full_path = get_path(pathroot, audio_path) if pathroot is not None else audio_path
    y = load_audio(full_path, sr=feature_cfg.sr)
    return build_feature_vector(
        y=y,
        feature_cfg=feature_cfg,
        pooling_mode=pooling_mode,
        pooling_cfg=pooling_cfg,
        chunk_cfg=chunk_cfg,
    )


def extract_features_from_row(
    df,
    idx: int,
    feature_cfg: FeatureConfig,
    pooling_mode: Literal["global", "chunk"],
    pooling_cfg: PoolingConfig,
    chunk_cfg: ChunkConfig | None = None,
    pathroot: str = "train_audio_dir",
    filename_col: str = "filename",
) -> tuple[np.ndarray, list[str]]:
    """Load one audio file and return pooled features + names."""
    path = build_audio_path(
        df=df,
        idx=idx,
        pathroot=pathroot,
        filename_col=filename_col,
    )
    y = load_audio(path, sr=feature_cfg.sr)
    return build_feature_vector(
        y=y,
        feature_cfg=feature_cfg,
        pooling_mode=pooling_mode,
        pooling_cfg=pooling_cfg,
        chunk_cfg=chunk_cfg,
    )


def build_feature_matrix_from_df(
    df,
    feature_cfg: FeatureConfig,
    pooling_mode: Literal["global", "chunk"],
    pooling_cfg: PoolingConfig,
    chunk_cfg: ChunkConfig | None = None,
    indices: list[int] | None = None,
    pathroot: str = "train_audio_dir",
    filename_col: str = "filename",
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
            feature_cfg=feature_cfg,
            pooling_mode=pooling_mode,
            pooling_cfg=pooling_cfg,
            chunk_cfg=chunk_cfg,
            pathroot=pathroot,
            filename_col=filename_col,
        )

        if pooling_mode == "global":
            if feature_names_ref is None:
                feature_names_ref = names
            elif names != feature_names_ref:
                raise RuntimeError(
                    "Inconsistent feature names across rows. "
                    "Ensure all rows produce vectors with the same schema."
                )

            vectors.append(np.asarray(vec, dtype=float).ravel())
            continue

        if pooling_mode != "chunk":
            raise ValueError(f"Unsupported pooling mode: {pooling_mode}")

        if chunk_cfg is None:
            raise ValueError("chunk_cfg is required for chunk mode")

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
    label_cols: str | list[str],
    feature_cfg: FeatureConfig,
    pooling_mode: Literal["global", "chunk"],
    pooling_cfg: PoolingConfig,
    chunk_cfg: ChunkConfig | None = None,
    indices: list[int] | None = None,
    pathroot: str = "train_audio_dir",
    filename_col: str = "filename",
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
            feature_cfg=feature_cfg,
            pooling_mode=pooling_mode,
            pooling_cfg=pooling_cfg,
            chunk_cfg=chunk_cfg,
            pathroot=pathroot,
            filename_col=filename_col,
        )

        if pooling_mode == "global":
            row_matrix = np.asarray(vec, dtype=float).ravel()[np.newaxis, :]
        elif pooling_mode == "chunk":
            row_matrix = np.asarray(vec, dtype=float)
            if row_matrix.ndim == 1:
                row_matrix = row_matrix[np.newaxis, :]
            if row_matrix.ndim != 2:
                raise RuntimeError("Chunk features must be a 2D matrix")
        else:
            raise ValueError(f"Unsupported pooling mode: {pooling_mode}")

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
