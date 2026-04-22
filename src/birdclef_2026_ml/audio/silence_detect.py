import librosa
import numpy as np


def compute_rms_dbfs(y, sr, frame_length, hop_length, ref_for_db):
    # Convert to dBFS (decibels relative to full scale)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    db = librosa.amplitude_to_db(rms, ref=ref_for_db)
    times = times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    return rms, db, times


def get_perc_threshold(rms_db, perc=10):
    # Assumption : at least perc% of the audio is silence
    return np.percentile(rms_db, perc)


def get_low_energy_median(rms_db, perc=15):
    # Focus on low-energy segments
    low = rms_db[rms_db < np.percentile(rms_db, perc)]
    return np.median(low)


def get_low_energy_median_band(rms_db, low_perc=5, high_perc=15):
    # Focus on low-energy segments
    low = rms_db[(rms_db > np.percentile(rms_db, low_perc)) & (rms_db < np.percentile(rms_db, high_perc))]
    return np.median(low)


def get_hysteresis(threshold, db_below):
    high_th = threshold
    low_th = threshold + db_below
    return high_th, low_th


def select_silence_frames_from_rms_db(
    rms_db,
    times,
    silence_th=-40.0,
    high_th=None,
    low_th=None,
):
    """Return silence mask and silence-frame seconds from RMS dB.

    If high_th and low_th are provided, hysteresis is used:
    - enter silence when rms_db <= high_th
    - leave silence when rms_db >= low_th
    """
    rms_db = np.asarray(rms_db, dtype=float)
    times = np.asarray(times, dtype=float)

    if high_th is None or low_th is None:
        silent_mask = rms_db <= silence_th
    else:
        silent_mask = np.zeros_like(rms_db, dtype=bool)
        in_silence = rms_db[0] <= high_th
        for i, db in enumerate(rms_db):
            if in_silence and db >= low_th:
                in_silence = False
            elif (not in_silence) and db <= high_th:
                in_silence = True
            silent_mask[i] = in_silence

    silent_segments = _silence_segments_from_mask(silent_mask, times)
    return silent_mask, times[silent_mask], silent_segments


def _silence_segments_from_mask(silent_mask, times):
    """Convert a boolean silence mask into contiguous (start_s, end_s) segments."""
    silent_mask = np.asarray(silent_mask, dtype=bool)
    times = np.asarray(times, dtype=float)

    if len(times) == 0:
        return []

    dt = np.median(np.diff(times)) if len(times) > 1 else 0.0
    padded = np.r_[False, silent_mask, False]
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1

    return [(times[s], times[e] + dt) for s, e in zip(starts, ends)]
