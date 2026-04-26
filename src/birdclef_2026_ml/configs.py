
# Spectral gating config dataclass
from typing import Optional, Literal
from birdclef_2026_ml.constants import SAMPLE_RATE
from typing import Literal
from dataclasses import dataclass, field
from dataclasses import dataclass

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
    n_mfcc: int = 20
    roll_percent: float = 0.85
    n_mels: int = 128  # balance between resolution & smoothing
    include_waveform_stats: bool = False
    include_mel_spectrogram: bool = True
    include_mfcc: bool = False
    include_delta: bool = True
    include_delta2: bool = True
    include_spectral: bool = True
    include_energy: bool = True

    def __post_init__(self):
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be > 0")
        if not (0.0 < self.roll_percent < 1.0):
            raise ValueError("roll_percent must be in (0, 1)")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be > 0")
        if self.n_mfcc <= 0:
            raise ValueError("n_mfcc must be > 0")


@dataclass(frozen=False)
class PoolingConfig:
    pooling_mode: Literal["global", "chunk"] = "chunk"
    stats: tuple[StatName, ...] = ("mean", "std", "min", "max")
    # percentiles: tuple[float, ...] = (10.0, 50.0, 90.0)
    percentiles: tuple[float, ...] = ()
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
