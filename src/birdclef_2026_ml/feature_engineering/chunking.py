
import numpy as np


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


# TODO: overestimates by 1 number of chunks for some audios
def count_nb_chunks(df, chunk_cfg, sr, hop_length):
    C = chunk_cfg.chunk_size_s
    S = chunk_cfg.step_size_s

    frame_rate_hz = feature_frame_rate_hz(sr, hop_length)
    # N = np.ceil(df["duration"] * frame_rate_hz).astype(int)
    # W = int(np.ceil(C * frame_rate_hz))
    # H = int(np.ceil(S * frame_rate_hz))

    # base = np.floor((N - W) / H).astype(int) + 1
    # base = np.maximum(1, base)  # handles short audio

    # last_end = (base - 1) * H + W
    # needs_extra = last_end < N

    # chunks = base + needs_extra.astype(int)
    # return np.sum(chunks)
    total = 0
    for dur in df["duration"].values:
        # n_frames = int(np.ceil(dur * frame_rate_hz))
        n_frames = 1 + np.floor(dur * sr / hop_length)
        intervals = get_sliding_window_intervals(
            n_frames,
            chunk_cfg.chunk_size_s,
            chunk_cfg.step_size_s,
            frame_rate_hz,
        )
        total += len(intervals)
    return total
