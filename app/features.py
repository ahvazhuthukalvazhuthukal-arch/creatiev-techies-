"""Forensic and acoustic feature extraction module for VocalGuard AI V2.

NOTE ON RESEARCH ARCHITECTURE (Creative Techies SIH PPT):
- Spectral flatness and spectral centroid provide signal distribution context.
- Linear Frequency Cepstral Coefficients (LFCC) and phase-spectrum statistics
  capture periodic vocoder framing and high-frequency spectral line artifacts.
- Biological vocal dynamics measures laryngeal vocal fold micro-jitter and tremor.
"""

from typing import Dict, Any
import numpy as np
import librosa
from scipy.fft import dct

def spectral_features(audio: np.ndarray, sr: int) -> dict:
    """Computes spectral flatness and spectral centroid statistics."""
    if len(audio) < 512:
        audio = np.pad(audio, (0, 512 - len(audio)), mode='constant')

    flatness = librosa.feature.spectral_flatness(y=audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    return {
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flatness_std": float(np.std(flatness)),
        "spectral_centroid_mean_hz": float(np.mean(centroid)),
        "spectral_centroid_std_hz": float(np.std(centroid)),
    }

def lfcc_features(audio: np.ndarray, sr: int, n_lfcc: int = 20) -> dict:
    """Extracts Linear Frequency Cepstral Coefficients (LFCC).

    LFCC uses linearly spaced triangular filterbanks (unlike Mel-scale filterbanks in MFCC)
    to capture high-frequency artifacts frequently introduced by neural vocoders and synthesis.
    EXPERIMENTAL: Features only, not a trained classifier.
    """
    if len(audio) < 512:
        audio = np.pad(audio, (0, 512 - len(audio)), mode='constant')

    stft = np.abs(librosa.stft(audio, n_fft=512, hop_length=160, win_length=400))
    power = stft ** 2
    n_filters = 20
    freqs = np.linspace(0, sr / 2, n_filters + 2)
    fft_freqs = np.linspace(0, sr / 2, power.shape[0])
    filterbank = np.zeros((n_filters, power.shape[0]))

    for i in range(n_filters):
        left, center, right = freqs[i], freqs[i + 1], freqs[i + 2]
        a = (fft_freqs >= left) & (fft_freqs <= center)
        b = (fft_freqs >= center) & (fft_freqs <= right)
        if center > left:
            filterbank[i, a] = (fft_freqs[a] - left) / (center - left)
        if right > center:
            filterbank[i, b] = (right - fft_freqs[b]) / (right - center)

    energies = np.maximum(np.dot(filterbank, power), 1e-10)
    lfcc = dct(np.log(energies), type=2, axis=0, norm="ortho")[:n_lfcc]

    return {
        "lfcc_mean": float(np.mean(lfcc)),
        "lfcc_std": float(np.std(lfcc)),
        "lfcc_abs_mean": float(np.mean(np.abs(lfcc))),
    }

def phase_features(audio: np.ndarray, sr: int) -> dict:
    """Extracts phase spectrum statistics from the STFT complex representation.

    Synthetic vocoders often generate discontinuous phase trajectories or artificial phase
    coherence compared to human vocal cords.
    EXPERIMENTAL: Features only, not a trained classifier.
    """
    if len(audio) < 512:
        audio = np.pad(audio, (0, 512 - len(audio)), mode='constant')

    stft = librosa.stft(audio, n_fft=512, hop_length=160, win_length=400)
    phase = np.angle(stft)
    unwrapped = np.unwrap(phase, axis=1)
    diff = np.diff(unwrapped, axis=1) if unwrapped.shape[1] > 1 else np.zeros_like(unwrapped)

    return {
        "phase_mean": float(np.mean(phase)),
        "phase_std": float(np.std(phase)),
        "phase_diff_std": float(np.std(diff)),
    }

def biological_vocal_dynamics(audio: np.ndarray, sr: int) -> dict:
    """Extracts biological vocal fold micro-tremor and pitch jitter dynamics.

    Human speech exhibits organic glottal micro-perturbations (0.5% - 2.5% jitter).
    Mechanical TTS synthesis exhibits rigid pitch quantization (< 0.2% jitter).
    """
    if len(audio) < 2048:
        return {"pitch_jitter": 0.005}

    try:
        f0 = librosa.yin(audio, fmin=60, fmax=400, sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 2:
            mean_f0 = float(np.mean(f0_valid))
            if mean_f0 > 10.0:
                jitter = float(np.mean(np.abs(np.diff(f0_valid))) / mean_f0)
            else:
                jitter = 0.005
        else:
            jitter = 0.005
    except Exception:
        jitter = 0.005

    return {"pitch_jitter": round(jitter, 6)}

def extract_all_features(audio: np.ndarray, sr: int) -> dict:
    """Extracts all acoustic, spectral, LFCC, phase, and biological features in one pass."""
    out = {}
    out.update(spectral_features(audio, sr))
    out.update(lfcc_features(audio, sr))
    out.update(phase_features(audio, sr))
    out.update(biological_vocal_dynamics(audio, sr))
    return out
