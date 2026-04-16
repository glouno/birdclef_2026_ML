from dataclasses import dataclass, field
from typing import Literal
import librosa
import numpy as np

from birdclef_2026_ml.constants import SAMPLE_RATE
from birdclef_2026_ml.audio_utils import build_audio_path, get_path, load_audio
StatName = Literal["mean", "std", "min", "max", "skew", "kurtosis"]


@dataclass(frozen=False)
class FeatureConfig:
    sr: int = SAMPLE_RATE
    n_fft: int = 1024
    hop_length: int = 512
    n_mfcc: int = 20
    roll_percent: float = 0.85
    include_waveform_stats: bool = False
    include_mfcc: bool = True
    include_delta: bool = True
    include_delta2: bool = True
    include_spectral: bool = True
    include_energy: bool = True

    def __post_init__(self):
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be > 0")
        if not (0.0 < self.roll_percent < 1.0):
            raise ValueError("roll_percent must be in (0, 1)")
        if self.n_mfcc <= 0:
            raise ValueError("n_mfcc must be > 0")


@dataclass(frozen=False)
class PoolingConfig:
    stats: tuple[StatName, ...] = ("mean", "std", "min", "max")
    percentiles: tuple[float, ...] = (10.0, 50.0, 90.0)
    nan_fill_value: float = 0.0

    def __post_init__(self):
        if len(set(self.stats)) != len(self.stats):
            raise ValueError("stats must be unique")
        for p in self.percentiles:
            if not (0.0 <= p <= 100.0):
                raise ValueError("percentiles must be in [0, 100]")


@dataclass(frozen=False)
class ChunkConfig:
    n_chunks: int = 5

    def __post_init__(self):
        if self.n_chunks <= 0:
            raise ValueError("n_chunks must be > 0")


@dataclass(frozen=False)
class PipelineConfig:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    pooling_mode: Literal["global", "chunk"] = "global"


# Feature extraction
def extract_mfcc_features(y: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    """Extract MFCC frame features and optional deltas from a waveform.

    Returns keys among: `mfcc`, `mfcc_delta`, `mfcc_delta2` depending on
    `FeatureConfig.include_*` flags.
    """
    if not (cfg.include_mfcc or cfg.include_delta or cfg.include_delta2):
        return {}

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )

    features: dict[str, np.ndarray] = {}
    if cfg.include_mfcc:
        features["mfcc"] = mfcc
    if cfg.include_delta:
        features["mfcc_delta"] = librosa.feature.delta(mfcc, order=1)
    if cfg.include_delta2:
        features["mfcc_delta2"] = librosa.feature.delta(mfcc, order=2)

    return features


def extract_spectral_features(y: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    """Extract frame-level spectral features from a waveform."""
    if not cfg.include_spectral:
        return {}

    features: dict[str, np.ndarray] = {
        "spectral_centroid": librosa.feature.spectral_centroid(
            y=y,
            sr=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(
            y=y,
            sr=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
        "spectral_rolloff": librosa.feature.spectral_rolloff(
            y=y,
            sr=cfg.sr,
            roll_percent=cfg.roll_percent,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
        "spectral_contrast": librosa.feature.spectral_contrast(
            y=y,
            sr=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
    }

    return features


def extract_energy_features(y: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    """Extract frame-level energy features from a waveform."""
    if not cfg.include_energy:
        return {}

    features: dict[str, np.ndarray] = {
        "rms": librosa.feature.rms(
            y=y,
            frame_length=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
        "zcr": librosa.feature.zero_crossing_rate(
            y,
            frame_length=cfg.n_fft,
            hop_length=cfg.hop_length,
        ),
    }

    return features


def extract_all_frame_features(
    y: np.ndarray,
    cfg: FeatureConfig,
) -> dict[str, np.ndarray]:
    """Extract all enabled frame-level features in deterministic order."""
    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.size == 0:
        raise ValueError("y must contain at least one sample")

    features: dict[str, np.ndarray] = {}

    if cfg.include_waveform_stats:
        features["waveform"] = y_arr[np.newaxis, :]

    features.update(extract_mfcc_features(y_arr, cfg))
    features.update(extract_spectral_features(y_arr, cfg))
    features.update(extract_energy_features(y_arr, cfg))
    return features

# Pooling


def _nan_skew(x: np.ndarray, axis: int = -1) -> np.ndarray:
    mean = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True)
    centered = x - mean
    m3 = np.nanmean(centered ** 3, axis=axis)
    s3 = np.squeeze(std, axis=axis) ** 3
    return np.where(s3 == 0.0, 0.0, m3 / s3)


def _nan_kurtosis(x: np.ndarray, axis: int = -1) -> np.ndarray:
    mean = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True)
    centered = x - mean
    m4 = np.nanmean(centered ** 4, axis=axis)
    s4 = np.squeeze(std, axis=axis) ** 4
    return np.where(s4 == 0.0, -3.0, m4 / s4 - 3.0)


def global_pool(feature_matrix: np.ndarray, pooling_cfg: PoolingConfig) -> np.ndarray:
    """Pool a feature matrix of shape (n_features, n_frames) into 1 vector."""
    x = np.asarray(feature_matrix, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2:
        raise ValueError("feature_matrix must be 1D or 2D")

    pieces: list[np.ndarray] = []
    for stat in pooling_cfg.stats:
        if stat == "mean":
            arr = np.nanmean(x, axis=1)
        elif stat == "std":
            arr = np.nanstd(x, axis=1)
        elif stat == "min":
            arr = np.nanmin(x, axis=1)
        elif stat == "max":
            arr = np.nanmax(x, axis=1)
        elif stat == "skew":
            arr = _nan_skew(x, axis=1)
        elif stat == "kurtosis":
            arr = _nan_kurtosis(x, axis=1)
        else:
            raise ValueError(f"Unsupported stat: {stat}")
        pieces.append(np.asarray(arr, dtype=float).ravel())

    if pooling_cfg.percentiles:
        p = np.asarray(pooling_cfg.percentiles, dtype=float)
        pct = np.nanpercentile(x, p, axis=1).T.reshape(-1)
        pieces.append(np.asarray(pct, dtype=float))

    if not pieces:
        return np.array([], dtype=float)

    out = np.concatenate(pieces)
    out = np.nan_to_num(out, nan=pooling_cfg.nan_fill_value)
    return out


def chunk_pool(
    feature_matrix: np.ndarray,
    chunk_cfg: ChunkConfig,
    pooling_cfg: PoolingConfig,
) -> np.ndarray:
    """Pool features into a fixed number of chunks and concatenate vectors."""
    x = np.asarray(feature_matrix, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2:
        raise ValueError("feature_matrix must be 1D or 2D")

    n_frames = x.shape[1]
    if n_frames == 0:
        return np.array([], dtype=float)

    if chunk_cfg.n_chunks > n_frames:
        raise ValueError(
            "n_chunks cannot be greater than the number of available frames "
            f"({chunk_cfg.n_chunks} > {n_frames})."
        )

    n_chunks = chunk_cfg.n_chunks
    bounds = np.linspace(0, n_frames, num=n_chunks + 1, dtype=int)

    chunks: list[np.ndarray] = []
    for idx in range(n_chunks):
        start = bounds[idx]
        end = bounds[idx + 1]
        if end <= start:
            continue
        chunk_vec = global_pool(x[:, start:end], pooling_cfg)
        chunks.append(chunk_vec)

    if not chunks:
        return np.array([], dtype=float)
    return np.concatenate(chunks)


def pool_feature_dict(
    features_dict: dict[str, np.ndarray],
    pooling_cfg: PoolingConfig,
    mode: Literal["global", "chunk"],
    chunk_cfg: ChunkConfig | None = None,
) -> dict[str, np.ndarray]:
    """Pool each feature matrix in a dict using global or chunk mode."""
    pooled: dict[str, np.ndarray] = {}

    if mode == "global":
        for name, mat in features_dict.items():
            pooled[name] = global_pool(mat, pooling_cfg)
        return pooled

    if mode == "chunk":
        if chunk_cfg is None:
            raise ValueError("chunk_cfg is required for chunk mode")
        for name, mat in features_dict.items():
            pooled[name] = chunk_pool(
                feature_matrix=mat,
                chunk_cfg=chunk_cfg,
                pooling_cfg=pooling_cfg,
            )
        return pooled

    raise ValueError(f"Unsupported pooling mode: {mode}")


def _percentile_label(percentile: float) -> str:
    p = float(percentile)
    if p.is_integer():
        return str(int(p))
    return str(p).replace(".", "_")


def get_feature_names(
    features_dict: dict[str, np.ndarray],
    pooling_cfg: PoolingConfig,
    pooling_mode: Literal["global", "chunk"],
    chunk_cfg: ChunkConfig | None = None,
) -> list[str]:
    """Return feature names in the exact same order as pooled vectors."""
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
            for chunk_idx in range(1, chunk_cfg.n_chunks + 1):
                names.extend([f"chunk{chunk_idx:02d}_{name}" for name in base_names])
        else:
            raise ValueError(f"Unsupported pooling mode: {pooling_mode}")

    return names

# Notebooks


def build_feature_vector(
    y: np.ndarray,
    feature_cfg: FeatureConfig,
    pooling_mode: Literal["global", "chunk"],
    pooling_cfg: PoolingConfig,
    chunk_cfg: ChunkConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build one pooled feature vector and aligned feature names."""
    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.size == 0:
        raise ValueError("y must contain at least one sample")

    frame_features = extract_all_frame_features(y_arr, feature_cfg)
    pooled = pool_feature_dict(
        features_dict=frame_features,
        pooling_cfg=pooling_cfg,
        mode=pooling_mode,
        chunk_cfg=chunk_cfg,
    )

    vectors: list[np.ndarray] = [np.asarray(v, dtype=float).ravel() for v in pooled.values()]
    if vectors:
        feature_vector = np.concatenate(vectors)
    else:
        feature_vector = np.array([], dtype=float)

    feature_names = get_feature_names(
        features_dict=frame_features,
        pooling_cfg=pooling_cfg,
        pooling_mode=pooling_mode,
        chunk_cfg=chunk_cfg,
    )

    if feature_vector.size != len(feature_names):
        raise RuntimeError(
            "Feature vector size and feature name count mismatch "
            f"({feature_vector.size} != {len(feature_names)})."
        )

    return feature_vector, feature_names


def extract_features_from_path(
    audio_path: str,
    feature_cfg: FeatureConfig,
    pooling_mode: Literal["global", "chunk"],
    pooling_cfg: PoolingConfig,
    chunk_cfg: ChunkConfig | None = None,
    pathroot: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load one audio file and return pooled feature vector + names."""
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
    """Load one audio file and return pooled feature vector + names."""
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
    """Build a 2D feature matrix by extracting one vector per dataframe row."""
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

        if feature_names_ref is None:
            feature_names_ref = names
        elif names != feature_names_ref:
            raise RuntimeError(
                "Inconsistent feature names across rows. "
                "Ensure all rows produce vectors with the same schema."
            )

        vectors.append(np.asarray(vec, dtype=float).ravel())

    if not vectors:
        return np.empty((0, 0), dtype=float), []

    feature_matrix = np.vstack(vectors)
    return feature_matrix, (feature_names_ref or [])
