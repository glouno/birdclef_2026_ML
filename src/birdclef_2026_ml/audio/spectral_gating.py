import numpy as np
import librosa
from scipy.ndimage import median_filter


def spectral_gating_snr(
    y,
    n_fft: int,
    hop_length: int,
    noise_percentile: float = 15,
    alpha: float = 1.5,
    smooth_freq: int = 3,
    smooth_time: int = 3,
    eps: float = 1e-8
):

    # 1. STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(S)
    phase = np.exp(1j * np.angle(S))

    # 2. Frame energy (RMS proxy)
    frame_energy = np.mean(magnitude**2, axis=0)

    # 3. Select low-energy frames (noise candidates)
    energy_thresh = np.percentile(frame_energy, noise_percentile)
    noise_frames = frame_energy <= energy_thresh

    # Too few noisy frames, avoid spectral gating entirely
    if np.sum(noise_frames) < 5:
        return y, S, None, None

    # 4. Frequency-dependent noise estimate N(t)
    noise_spectrum = np.median(
        magnitude[:, noise_frames],
        axis=1,
        keepdims=True
    )

    # 5. Soft SNR-style mask (approximating Wiener)
    snr = magnitude / (noise_spectrum + eps)
    mask = snr / (snr + alpha)

    # 6. Smoothing (critical for avoiding musical noise)
    mask = median_filter(mask, size=(smooth_freq, smooth_time))

    # 7. Apply mask
    S_clean = S * mask

    # 8. Reconstruct
    y_clean = librosa.istft(S_clean, hop_length=hop_length, length=len(y))

    return y_clean, S, S_clean, mask
