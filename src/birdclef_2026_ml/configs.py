
# Spectral gating config dataclass
from pathlib import Path
import yaml
from typing import Any, Literal
from dataclasses import asdict, dataclass, field

from birdclef_2026_ml.constants import SAMPLE_RATE
from birdclef_2026_ml.paths import load_project_paths

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
    mil_mode: bool = False


@dataclass(frozen=True)
class ModelConfig:
    type: Literal["SGDClassifier"] = "SGDClassifier"
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
    epochs: int = 5
    train_val_split: bool = True
    scope: bool = False

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


def build_pipeline_config(payload: dict[str, Any] | None = None) -> PipelineConfig:
    data = payload or {}
    return PipelineConfig(
        mil_mode=data.get("mil_mode", False),
        feature=FeatureConfig(**data.get("feature", {})),
        pooling=PoolingConfig(**data.get("pooling", {})),
        chunk=ChunkConfig(**data.get("chunk", {})),
        mil=MILConfig(**data.get("mil", {}))
    )


def build_experiment_config(payload: dict[str, Any] | None = None) -> ExperimentConfig:
    data = payload or {}
    return ExperimentConfig(
        model=ModelConfig(**data.get("model", {})),
        training=TrainingConfig(**data.get("training", {})),
    )


def build_calibration_config(payload: dict[str, Any] | None = None) -> CalibrationConfig:
    data = payload or {}
    calibration_payload = data.get("calibration", data)
    return CalibrationConfig(
        type=calibration_payload.get("type", "CalibratedClassifierCV"),
        params=calibration_payload.get("params", {"method": "sigmoid"}),
    )


def load_pipeline_config(run_name: str) -> PipelineConfig:
    paths = load_project_paths()
    config_path = paths.experiment_pipeline_path(run_name)

    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return build_pipeline_config(data)


def load_experiment_config(run_name: str, experiment_name: str) -> tuple[ExperimentConfig, Path]:
    paths = load_project_paths()
    config_path = paths.experiment_config_path(run_name, experiment_name)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment config not found: {config_path}. "
            f"Create {experiment_name}.yaml under {config_path.parent}."
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return build_experiment_config(payload), config_path


def load_calibration_config(run_name: str, calibration_name: str) -> tuple[CalibrationConfig, Path]:
    paths = load_project_paths()
    config_path = paths.experiments_dir / run_name / f"{calibration_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Calibration config not found: {config_path}. "
            f"Create {calibration_name}.yaml under {config_path.parent}."
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return build_calibration_config(payload), config_path
