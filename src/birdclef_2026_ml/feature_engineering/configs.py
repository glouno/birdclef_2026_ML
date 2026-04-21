
from typing import Optional
from dataclasses import dataclass, field
from typing import Literal

from birdclef_2026_ml.constants import SAMPLE_RATE

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
    chunk_size_s: float = 5.0     # seconds
    step_size_s: float = 5.0  # seconds

    def __post_init__(self):
        if self.chunk_size_s <= 0 or self.step_size_s <= 0:
            raise ValueError("chunks_s and step_size_s must be > 0")


@dataclass(frozen=False)
class PipelineConfig:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    pooling_mode: Literal["global", "chunk"] = "global"


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
