import numpy as np


def _pad_short_audio_randomly(y: np.ndarray, min_samples: int) -> np.ndarray:
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

    return np.pad(y, (pad_left, pad_right), mode="constant", constant_values=0.0)


def _percentile_label(percentile: float) -> str:
    p = float(percentile)
    if p.is_integer():
        return str(int(p))
    return str(p).replace(".", "_")
