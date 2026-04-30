
# Spectral gating config dataclass
from typing import Any, Literal
from dataclasses import asdict, dataclass, field

from birdclef_2026_ml.constants import SAMPLE_RATE

StatName = Literal["mean", "std", "min", "max", "skew", "kurtosis"]


@dataclass(frozen=False)
class SpectralGatingConfig:
    sr: int
    n_fft: int
    hop_length: int
    noise_percentile: float
    alpha: float
    smooth_freq: int
    smooth_time: int
    eps: float


@dataclass(frozen=False)
class FeatureConfig:
    sr: int = SAMPLE_RATE
    n_fft: int = 1024
    hop_length: int = 512
    roll_percent: float = 0.85
    n_mels: int = 128
    include_mel_spectrogram: bool = True
    include_delta: bool = True
    include_delta2: bool = True
    include_spectral: bool = True
    include_energy: bool = True
    include_texture: bool = True
    texture_downsample_freq_bins: int = 32
    texture_quant_levels: int = 8
    texture_patch_frames: int = 9
    lbp_n_bins: int = 8
    event_window_frames: int = 9
    event_threshold_std: float = 1.0

    def __post_init__(self):
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be > 0")
        if not (0.0 < self.roll_percent < 1.0):
            raise ValueError("roll_percent must be in (0, 1)")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be > 0")
        if self.texture_downsample_freq_bins <= 0:
            raise ValueError("texture_downsample_freq_bins must be > 0")
        if self.texture_quant_levels <= 1:
            raise ValueError("texture_quant_levels must be > 1")
        if self.texture_patch_frames < 3:
            raise ValueError("texture_patch_frames must be >= 3")
        if self.lbp_n_bins <= 0:
            raise ValueError("lbp_n_bins must be > 0")
        if self.event_window_frames <= 0:
            raise ValueError("event_window_frames must be > 0")


@dataclass(frozen=False)
class PoolingConfig:
    pooling_mode: Literal["global", "chunk"] = "chunk"
    # stats: tuple[StatName, ...] = ("mean", "std", "min", "max")
    # percentiles: tuple[float, ...] = (10.0, 50.0, 90.0)
    stats: tuple[StatName, ...] = ("mean", "std")
    percentiles: tuple[float, ...] = (10.0, 90.0)
    nan_fill_value: float = 0.0

    def __post_init__(self):
        if len(set(self.stats)) != len(self.stats):
            raise ValueError("stats must be unique")
        for p in self.percentiles:
            if not (0.0 <= p <= 100.0):
                raise ValueError("percentiles must be in [0, 100]")


@dataclass(frozen=False)
class ChunkConfig:
    chunk_size_s: float = 5.0     # seconds
    step_size_s: float = 5.0  # seconds

    def __post_init__(self):
        if self.chunk_size_s <= 0 or self.step_size_s <= 0:
            raise ValueError("chunks_s and step_size_s must be > 0")


@dataclass(frozen=False)
class MILConfig:
    window_size_s: float = 1.0  # seconds
    step_size_s: float = 0.5    # seconds
    pooling: Literal["max", "mean", "logsumexp"] = "max"

    def __post_init__(self):
        if self.window_size_s <= 0 or self.step_size_s <= 0:
            raise ValueError("window_size_s and step_size_s must be > 0")
        if self.pooling not in ("max", "mean", "logsumexp"):
            raise ValueError("Invalid pooling method for MILConfig")


# TODO: handle edge cases here (e.g. pooling = chunk but chunk_cfg is not passed)
@dataclass(frozen=False)
class PipelineConfig:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    mil: MILConfig = field(default_factory=MILConfig)


@dataclass(frozen=True)
class ModelConfig:
    type: Literal["SGDClassifier", "LogisticRegression"] = "SGDClassifier"
    params: dict[str, Any] = field(default_factory=lambda: {
        "loss": "log_loss",
        "penalty": "elasticnet",
        "alpha": 1e-4,
        "max_iter": 3000,
        "tol": 1e-4,
        "early_stopping": False,
        "learning_rate": "optimal",
        "average": True,
        "random_state": 42,
    })

    def __post_init__(self):
        if not self.type:
            raise ValueError("model.type must be set")
        if not isinstance(self.params, dict):
            raise TypeError("model.params must be a mapping")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 3
    train_val_split: bool = False

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be > 0")
        if self.epochs <= 0:
            raise ValueError("training.epochs must be > 0")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_experiment_config(payload: dict[str, Any] | None = None) -> ExperimentConfig:
    data = payload or {}
    return ExperimentConfig(
        model=ModelConfig(**data.get("model", {})),
        training=TrainingConfig(**data.get("training", {})),
    )


@dataclass(frozen=True)
class CalibrationConfig:
    type: Literal["CalibratedClassifierCV"] = "CalibratedClassifierCV"
    params: dict[str, Any] = field(default_factory=lambda: {
        "method": "sigmoid",
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self):
        if not self.type:
            raise ValueError("calibration.type must be set")
        if not isinstance(self.params, dict):
            raise TypeError("calibration.params must be a mapping")


def build_calibration_config(payload: dict[str, Any] | None = None) -> CalibrationConfig:
    data = payload or {}
    calibration_payload = data.get("calibration", data)
    return CalibrationConfig(
        type=calibration_payload.get("type", "CalibratedClassifierCV"),
        params=calibration_payload.get("params", {"method": "sigmoid"}),
    )
