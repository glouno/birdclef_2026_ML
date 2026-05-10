
import numpy as np
from birdclef_2026_ml.configs import ChunkConfig, MILConfig


def feature_frame_rate_hz(sr, hop_length) -> float:
    # if feature_name == "waveform":
    #     return float(sr)
    # Frame_rate_hz for any feature except waveform
    return float(sr) / float(hop_length)


def _chunk_size_in_frames(chunks_s, frame_rate_hz: float) -> int:
    if frame_rate_hz <= 0.0:
        raise ValueError("frame_rate_hz must be > 0")
    return max(1, int(np.ceil(float(chunks_s) * frame_rate_hz)))


def _window_step_in_frames(step_size_s, frame_rate_hz: float) -> int:
    if frame_rate_hz <= 0.0:
        raise ValueError("frame_rate_hz must be > 0")
    if step_size_s <= 0:
        raise ValueError("step_size_s must be > 0")
    return max(1, int(np.ceil(float(step_size_s) * frame_rate_hz)))


def get_sliding_window_intervals(
    n_frames: int,
    window_size_s: float,
    step_size_s: float,
    frame_rate_hz: float,
) -> list[tuple[int, int]]:
    """Return sliding-window frame intervals for arbitrary window and step sizes."""
    window_size = _chunk_size_in_frames(window_size_s, frame_rate_hz)
    step = _window_step_in_frames(step_size_s, frame_rate_hz)
    if n_frames <= 0 or window_size > n_frames:
        return [(0, n_frames)] if n_frames > 0 else []
    windows: list[tuple[int, int]] = []
    start = 0
    while start + window_size < n_frames:
        end = start + window_size
        windows.append((start, end))
        start += step

    if not windows or windows[-1][1] < n_frames:
        end = n_frames
        start = max(0, end - window_size)
        windows.append((start, end))
    # print(windows)
    return windows

# Note : can overestimate the number of chunks if with_pad=False


def get_duration_chunk_intervals(
    duration_s: float,
    chunk_cfg: ChunkConfig | MILConfig,
    sr: int,
    n_fft: int,
    hop_length: int,
    with_pad: bool = False
) -> list[tuple[int, int]]:
    frame_rate_hz = feature_frame_rate_hz(sr, hop_length)
    if with_pad:
        pad = n_fft // 2
        n_frames = 1 + np.floor((duration_s * sr + 2*pad - n_fft) / hop_length)
    else:
        n_frames = int(1 + np.floor(duration_s * sr / hop_length))

    return get_sliding_window_intervals(
        n_frames=n_frames,
        window_size_s=chunk_cfg.chunk_size_s,
        step_size_s=chunk_cfg.step_size_s,
        frame_rate_hz=frame_rate_hz,
    )


# When counting with mil_mode, assumption that ChunkConfig.chunk_size_s = ChunkConfig.step_size_s
def count_nb_chunks(
    df,
    chunk_cfg: ChunkConfig | MILConfig,
    sr,
    n_fft: int,
    hop_length,
    with_pad: bool = False
):
    total = 0
    for dur in df["duration"].values:
        intervals = get_duration_chunk_intervals(dur, chunk_cfg, sr, n_fft, hop_length, with_pad)
        total += len(intervals)
    return total
