import numpy as np


def frame_energy(S, power=2.0, eps=1e-10):
    """
    Compute per-frame energy from spectrogram.

    S: (n_freq, n_frames)
    power: 1.0 for magnitude, 2.0 for power
    """
    if power == 1.0:
        E = np.sum(np.abs(S), axis=0)
    else:
        E = np.sum(S**2, axis=0)
    return E + eps


def normalize_weights(w):
    w = np.asarray(w)
    w = w / (np.sum(w) + 1e-12)
    return w


def weighted_percentile(values, weights, q):
    """
    Compute weighted percentile along last axis.
    values: (n_freq, n_frames)
    weights: (n_frames,)
    q: percentile in [0, 100]
    """

    sorter = np.argsort(values, axis=1)
    values_sorted = np.take_along_axis(values, sorter, axis=1)
    weights_sorted = np.take_along_axis(
        np.broadcast_to(weights, values.shape), sorter, axis=1
    )

    cdf = np.cumsum(weights_sorted, axis=1)
    cutoff = q / 100.0

    # find first index where cdf >= cutoff
    idx = (cdf >= cutoff).argmax(axis=1)

    return values_sorted[np.arange(values.shape[0]), idx]


def weighted_median(S, weights):
    return weighted_percentile(S, weights, q=50)


def weighted_trimmed_mean(S, weights, trim_ratio=0.1):
    """
    Trim low and high tails based on weighted percentiles.
    """
    low = weighted_percentile(S, weights, q=100 * trim_ratio)
    high = weighted_percentile(S, weights, q=100 * (1 - trim_ratio))

    # mask values outside [low, high]
    mask = (S >= low[:, None]) & (S <= high[:, None])

    w = weights[None, :] * mask
    w = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)

    return np.sum(S * w, axis=1)


def weighted_mean(S, weights):
    return np.sum(S * weights[None, :], axis=1)


def robust_time_pooling(
    S,
    method="median",
    percentile=80,
    trim_ratio=0.1,
    energy_power=2.0,
):
    """
    Main entry point.

    S: (n_freq, n_frames)
    method: "median" | "percentile" | "trimmed_mean" | "mean"
    """
    E = frame_energy(S, power=energy_power)
    w = normalize_weights(E)

    if method == "median":
        return weighted_median(S, w)

    elif method == "percentile":
        return weighted_percentile(S, w, q=percentile)

    elif method == "trimmed_mean":
        return weighted_trimmed_mean(S, w, trim_ratio=trim_ratio)

    elif method == "mean":
        return weighted_mean(S, w)

    else:
        raise ValueError(f"Unknown method: {method}")
