from birdclef_2026_ml.feature_engineering.configs import SpectralGatingConfig
import numpy as np
import librosa
from scipy.ndimage import median_filter


def spectral_gating_snr(
    y,
    config: SpectralGatingConfig
):

    # 1. STFT
    S = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)
    magnitude = np.abs(S)
    phase = np.exp(1j * np.angle(S))

    # 2. Frame energy (RMS proxy)
    frame_energy = np.mean(magnitude**2, axis=0)

    # 3. Select low-energy frames (noise candidates)
    energy_thresh = np.percentile(frame_energy, config.noise_percentile)
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
    snr = magnitude / (noise_spectrum + config.eps)
    mask = snr / (snr + config.alpha)

    # 6. Smoothing (critical for avoiding musical noise)
    mask = median_filter(mask, size=(config.smooth_freq, config.smooth_time))

    # 7. Apply mask
    S_clean = S * mask

    # 8. Reconstruct
    y_clean = librosa.istft(S_clean, hop_length=config.hop_length, length=len(y))

    return y_clean, S, S_clean, mask
