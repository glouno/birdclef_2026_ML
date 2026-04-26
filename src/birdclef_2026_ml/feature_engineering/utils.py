import numpy as np
from pathlib import Path
import joblib


def pad_short_audio(y_arr, chunk_size_s, sr):
    min_samples = int(np.ceil(float(chunk_size_s) * float(sr)))
    return _pad_short_audio_randomly(y_arr, min_samples=min_samples)


def _pad_short_audio_randomly(y: np.ndarray, min_samples: int) -> np.ndarray:
    # Precaution to avoid unnecessary padding
    if y.size >= min_samples:
        return y

    pad_total = int(min_samples - y.size)
    where = np.random.choice(("begin", "end", "both"))

    if where == "begin":
        pad_left, pad_right = pad_total, 0
    elif where == "end":
        pad_left, pad_right = 0, pad_total
    else:
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

    y_pad = np.pad(y, (pad_left, pad_right), mode="constant", constant_values=0.0)
    return y_pad


def _percentile_label(percentile: float) -> str:
    p = float(percentile)
    if p.is_integer():
        return str(int(p))
    return str(p).replace(".", "_")


def joblib_to_frame_features(
    payload: dict | str | Path,
) -> dict[str, np.ndarray]:
    """Convert stored joblib payload into extract_all_frame_features-like output."""
    loaded = joblib.load(payload) if isinstance(payload, (str, Path)) else payload
    if not isinstance(loaded, dict):
        raise TypeError("Expected a dict payload for precomputed frame features.")

    out: dict[str, np.ndarray] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise TypeError("Feature names in precomputed payload must be strings.")
        out[key] = np.asarray(value, dtype=float)
    return out
