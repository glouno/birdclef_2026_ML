
import numpy as np


def feature_frame_rate_hz(feature_name: str, sr, hop_length) -> float:
    if feature_name == "waveform":
        return float(sr)
    return float(sr) / float(hop_length)


def _chunk_size_in_frames(chunks_s, frame_rate_hz: float) -> int:
    if frame_rate_hz <= 0.0:
        raise ValueError("frame_rate_hz must be > 0")
    return max(1, int(np.ceil(float(chunks_s) * frame_rate_hz)))


def _chunk_step_in_frames(chunks_s, overlap, frame_rate_hz: float) -> int:
    """Return the step size (stride) in frames between chunk starts, accounting for overlap."""
    chunk_size = _chunk_size_in_frames(chunks_s, frame_rate_hz)
    overlap_frames = int(np.round(overlap * frame_rate_hz))
    step = chunk_size - overlap_frames
    return max(1, step)


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
    print("windows", windows, n_frames, step_size_s)
    return windows


# def get_chunk_intervals(n_frames: int, chunks_s: float, step_size_s: float, frame_rate_hz: float) -> list[tuple[int, int]]:
#     """
#     Return a list of (start, end) frame indices for each chunk.
#     The last chunk always covers the last `chunks_s` duration, ignoring hop if necessary.
#     """
#     return get_sliding_window_intervals(
#         n_frames=n_frames,
#         window_size_s=chunks_s,
#         step_size_s=step_size_s,
#         frame_rate_hz=frame_rate_hz,
#     )
