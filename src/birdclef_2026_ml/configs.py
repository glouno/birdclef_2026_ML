
# Spectral gating config dataclass
from pathlib import Path
import yaml
import json
from typing import Any, Literal
from dataclasses import asdict, dataclass, field
from sklearn.linear_model import SGDClassifier

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
    chunk_size_s: float = 1.0  # seconds
    step_size_s: float = 0.5    # seconds
    pooling: Literal["max", "mean", "logsumexp"] = "max"

    def __post_init__(self):
        if self.chunk_size_s <= 0 or self.step_size_s <= 0:
            raise ValueError("chunk_size_s and step_size_s must be > 0")
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
    early_stopping: bool = False
    n_iter_no_change: int = 5
    tol: float = 1e-4

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be > 0")
        if self.epochs <= 0:
            raise ValueError("training.epochs must be > 0")
        if self.n_iter_no_change <= 0:
            raise ValueError("training.n_iter_no_change must be > 0")
        if self.tol < 0:
            raise ValueError("training.tol must be >= 0")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    second_stage: "SecondStageConfig" = field(default_factory=lambda: SecondStageConfig())

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


@dataclass(frozen=True)
class SecondStageModelConfig:
    type: Literal["ExplainableBoostingClassifier"] = "ExplainableBoostingClassifier"
    params: dict[str, Any] = field(default_factory=lambda: {
        "interactions": 0,
        "learning_rate": 0.03,
        "max_rounds": 500,
        "max_bins": 256,
        "max_interaction_bins": 64,
        "min_samples_leaf": 2,
        "outer_bags": 0,
        "inner_bags": 0,
        "validation_size": 0.0,
        "random_state": 42,
    })

    def __post_init__(self):
        if not self.type:
            raise ValueError("second_stage.model.type must be set")
        if not isinstance(self.params, dict):
            raise TypeError("second_stage.model.params must be a mapping")


@dataclass(frozen=True)
class SecondStageAugmentationConfig:
    include_clean_bags: bool = True
    include_soundscape_bags: bool = True
    synthetic_bag_multiplier: float = 2.0
    max_mixture_size: int = 4
    mixture_size_probabilities: dict[int, float] = field(default_factory=dict)
    mixture_method: Literal["weighted", "max"] = "weighted"
    weight_min: float = 0.2
    weight_max: float = 1.0
    temporal_mask_prob: float = 1.0
    temporal_keep_ratio_min: float = 0.3
    temporal_keep_ratio_max: float = 0.7
    temporal_mask_min_segments: int = 1
    temporal_mask_max_segments: int = 2
    temporal_fill_value: Literal["zero", "prior"] = "prior"
    gaussian_noise_std: float = 0.03
    random_state: int = 42

    def __post_init__(self):
        if self.synthetic_bag_multiplier < 0:
            raise ValueError("second_stage.augmentation.synthetic_bag_multiplier must be >= 0")
        if self.max_mixture_size <= 0:
            raise ValueError("second_stage.augmentation.max_mixture_size must be > 0")
        if self.mixture_method not in ("weighted", "max"):
            raise ValueError("second_stage.augmentation.mixture_method must be weighted or max")
        if self.weight_min <= 0 or self.weight_max <= 0:
            raise ValueError("second_stage.augmentation weights must be > 0")
        if self.weight_min > self.weight_max:
            raise ValueError("second_stage.augmentation.weight_min must be <= weight_max")
        if not (0.0 <= self.temporal_mask_prob <= 1.0):
            raise ValueError("second_stage.augmentation.temporal_mask_prob must be in [0, 1]")
        if not (0.0 < self.temporal_keep_ratio_min <= 1.0):
            raise ValueError(
                "second_stage.augmentation.temporal_keep_ratio_min must be in (0, 1]"
            )
        if not (0.0 < self.temporal_keep_ratio_max <= 1.0):
            raise ValueError(
                "second_stage.augmentation.temporal_keep_ratio_max must be in (0, 1]"
            )
        if self.temporal_keep_ratio_min > self.temporal_keep_ratio_max:
            raise ValueError(
                "second_stage.augmentation.temporal_keep_ratio_min must be <= "
                "temporal_keep_ratio_max"
            )
        if self.temporal_mask_min_segments <= 0 or self.temporal_mask_max_segments <= 0:
            raise ValueError("second_stage.augmentation temporal mask segments must be > 0")
        if self.temporal_mask_min_segments > self.temporal_mask_max_segments:
            raise ValueError(
                "second_stage.augmentation.temporal_mask_min_segments must be <= "
                "temporal_mask_max_segments"
            )
        if self.temporal_fill_value not in ("zero", "prior"):
            raise ValueError(
                "second_stage.augmentation.temporal_fill_value must be zero or prior"
            )
        if self.gaussian_noise_std < 0:
            raise ValueError("second_stage.augmentation.gaussian_noise_std must be >= 0")


@dataclass(frozen=True)
class SecondStageConfig:
    enabled: bool = False
    n_jobs: int | None = None
    model: SecondStageModelConfig = field(default_factory=SecondStageModelConfig)
    augmentation: SecondStageAugmentationConfig = field(default_factory=SecondStageAugmentationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        second_stage=build_second_stage_config(data.get("second_stage")),
    )


def build_calibration_config(payload: dict[str, Any] | None = None) -> CalibrationConfig:
    data = payload or {}
    calibration_payload = data.get("calibration", data)
    return CalibrationConfig(
        type=calibration_payload.get("type", "CalibratedClassifierCV"),
        params=calibration_payload.get("params", {"method": "sigmoid"}),
    )


def build_second_stage_config(payload: dict[str, Any] | None = None) -> SecondStageConfig:
    data = payload or {}
    return SecondStageConfig(
        enabled=bool(data.get("enabled", False)),
        n_jobs=data.get("n_jobs"),
        model=SecondStageModelConfig(**data.get("model", {})),
        augmentation=SecondStageAugmentationConfig(**data.get("augmentation", {})),
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


def artifact_stem(experiment_cfg: ExperimentConfig) -> str:
    return experiment_cfg.model.type.lower()


ESTIMATOR_REGISTRY = {
    "SGDClassifier": SGDClassifier,
}


def build_estimator(experiment_cfg: ExperimentConfig):
    estimator_cls = ESTIMATOR_REGISTRY.get(experiment_cfg.model.type)
    if estimator_cls is None:
        supported = ", ".join(sorted(ESTIMATOR_REGISTRY))
        raise ValueError(f"Unsupported model.type={experiment_cfg.model.type}. Supported: {supported}")
    params = dict(experiment_cfg.model.params)
    params.pop("early_stopping", None)
    params.pop("n_iter_no_change", None)
    params.pop("validation_fraction", None)
    return estimator_cls(**params)


def build_second_stage_estimator(second_stage_cfg: SecondStageConfig):
    if second_stage_cfg.model.type != "ExplainableBoostingClassifier":
        raise ValueError(
            "Unsupported second_stage.model.type="
            f"{second_stage_cfg.model.type}. Supported: ExplainableBoostingClassifier"
        )

    from interpret.glassbox import ExplainableBoostingClassifier

    return ExplainableBoostingClassifier(**dict(second_stage_cfg.model.params))


def build_calibration_params(calibration_cfg: CalibrationConfig) -> dict[str, object]:
    if calibration_cfg.type != "CalibratedClassifierCV":
        raise ValueError(
            f"Unsupported calibration.type={calibration_cfg.type}. "
            "Supported: CalibratedClassifierCV"
        )

    params = dict(calibration_cfg.params)
    params.pop("cv", None)
    params.pop("estimator", None)
    params.pop("n_jobs", None)
    return params


def load_config(path: Path):
    config = dict()
    with open(path) as f:
        config = json.load(f)
    return config


def save_config(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj.__dict__, f, indent=2)
