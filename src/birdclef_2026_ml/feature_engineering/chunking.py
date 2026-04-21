
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


# def _n_chunks_from_time(n_frames: int, chunk_cfg: ChunkConfig, frame_rate_hz: float) -> int:
#     if n_frames <= 0:
#         return 0
#     chunk_size_frames = _chunk_size_in_frames(chunk_cfg, frame_rate_hz)
#     step_frames = _chunk_step_in_frames(chunk_cfg, frame_rate_hz)
#     if n_frames < chunk_size_frames:
#         return 1
#     return 1 + max(0, (n_frames - chunk_size_frames) // step_frames)


def get_chunk_intervals(n_frames: int, chunks_s: float, overlap: float, frame_rate_hz: float) -> list[tuple[int, int]]:
    """
    Return a list of (start, end) frame indices for each chunk.
    The last chunk always covers the last `chunks_s` duration, ignoring hop if necessary.
    """
    chunk_size = max(1, int(np.ceil(chunks_s * frame_rate_hz)))
    step = max(1, int(np.ceil((chunks_s - overlap) * frame_rate_hz)))
    if n_frames <= 0 or chunk_size > n_frames:
        return [(0, n_frames)] if n_frames > 0 else []

    chunks = []
    start = 0
    while start + chunk_size < n_frames:
        end = start + chunk_size
        chunks.append((start, end))
        start += step
    # Always add the last chunk to cover the last chunk_size frames
    if not chunks or chunks[-1][1] < n_frames:
        end = n_frames
        start = max(0, end - chunk_size)
        chunks.append((start, end))
    return chunks
