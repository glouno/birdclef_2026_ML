import numpy as np


def pad_short_audio(y_arr, chunk_cfg, feature_cfg):
    min_samples = int(np.ceil(float(chunk_cfg.chunk_size_s) * float(feature_cfg.sr)))
    return _pad_short_audio_randomly(y_arr, min_samples=min_samples)


def _pad_short_audio_randomly(y: np.ndarray, min_samples: int) -> np.ndarray:
    print("(1) _pad_short_audio_randomly", y.size, min_samples)
    if y.size >= min_samples:
        return y
    print("(2) _pad_short_audio_randomly")

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
    print("(3) padded", y_pad)
    return y_pad


def _percentile_label(percentile: float) -> str:
    p = float(percentile)
    if p.is_integer():
        return str(int(p))
    return str(p).replace(".", "_")
