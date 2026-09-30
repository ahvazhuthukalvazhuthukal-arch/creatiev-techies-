"""Audio Quality Gate & Voice Activity Assessment Module.
Evaluates physical signal properties: RMS energy, clipping ratio, SNR estimation,
silence ratio, and zero-crossing rate to determine whether an audio window is suitable
for forensic neural evaluation.
"""
from typing import Dict, Any, List
import numpy as np


def compute_vad_and_energy(
    audio: np.ndarray,
    sr: int = 16000,
    frame_ms: float = 20.0,
    energy_threshold: float = 0.005,
    zcr_min: float = 0.01,
    zcr_max: float = 0.40,
) -> Dict[str, Any]:
    """Lightweight multi-feature Voice Activity Detection (VAD).
    Partitions window into frame_ms segments, evaluating RMS energy and Zero-Crossing Rate.
    """
    if len(audio) == 0:
        return {
            "speech_detected": False,
            "speech_ratio": 0.0,
            "rms_energy": 0.0,
            "peak_amplitude": 0.0,
            "voiced_frames": 0,
            "total_frames": 0,
        }

    frame_len = max(int(sr * (frame_ms / 1000.0)), 1)
    num_frames = len(audio) // frame_len
    
    if num_frames == 0:
        rms_val = float(np.sqrt(np.mean(audio ** 2)))
        peak_val = float(np.max(np.abs(audio)))
        is_speech = rms_val >= energy_threshold
        return {
            "speech_detected": bool(is_speech),
            "speech_ratio": 1.0 if is_speech else 0.0,
            "rms_energy": round(rms_val, 5),
            "peak_amplitude": round(peak_val, 5),
            "voiced_frames": 1 if is_speech else 0,
            "total_frames": 1,
        }

    voiced_count = 0
    frames = audio[:num_frames * frame_len].reshape(num_frames, frame_len)
    
    # Vectorized RMS across frames
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    
    # Vectorized Zero-Crossing Rate
    signs = np.sign(frames)
    signs[signs == 0] = 1
    zcr = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)

    for i in range(num_frames):
        if frame_rms[i] >= energy_threshold and (zcr_min <= zcr[i] <= zcr_max):
            voiced_count += 1

    overall_rms = float(np.sqrt(np.mean(audio ** 2)))
    overall_peak = float(np.max(np.abs(audio)))
    speech_ratio = float(voiced_count / num_frames) if num_frames > 0 else 0.0
    speech_detected = speech_ratio >= 0.20 or (overall_rms >= energy_threshold * 1.5)

    return {
        "speech_detected": bool(speech_detected),
        "speech_ratio": round(speech_ratio, 3),
        "rms_energy": round(overall_rms, 5),
        "peak_amplitude": round(overall_peak, 5),
        "voiced_frames": int(voiced_count),
        "total_frames": int(num_frames),
    }


def evaluate_audio_quality(
    audio: np.ndarray,
    sr: int = 16000,
) -> Dict[str, Any]:
    """Comprehensive Audio Quality Gate.
    Analyzes audio integrity and assigns a 0-100 quality score.
    If quality is below minimum threshold (< 40), flags as INCONCLUSIVE.
    """
    if len(audio) == 0:
        return {
            "quality_score": 0,
            "quality_tier": "POOR",
            "is_sufficient": False,
            "reasons": ["Empty audio buffer"],
            "metrics": {
                "duration_seconds": 0.0,
                "rms_energy": 0.0,
                "peak_amplitude": 0.0,
                "clipping_ratio": 0.0,
                "estimated_snr_db": 0.0,
                "silence_ratio": 1.0,
            },
        }

    # Sanitize NaN/Inf
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

    duration_sec = float(len(audio) / sr)
    rms_val = float(np.sqrt(np.mean(audio ** 2)))
    peak_val = float(np.max(np.abs(audio)))

    # 1. Clipping detection (samples >= 0.99)
    clipped_samples = int(np.sum(np.abs(audio) >= 0.99))
    clipping_ratio = float(clipped_samples / len(audio)) if len(audio) > 0 else 0.0

    # 2. Silence ratio (samples with energy < 0.003)
    frame_size = max(int(sr * 0.02), 1)  # 20ms frames
    num_frames = len(audio) // frame_size
    if num_frames > 0:
        frames = audio[:num_frames * frame_size].reshape(num_frames, frame_size)
        f_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        silent_frames = int(np.sum(f_rms < 0.003))
        silence_ratio = float(silent_frames / num_frames)
    else:
        silence_ratio = 1.0 if rms_val < 0.003 else 0.0

    # 3. Simple SNR estimation (peak signal power vs estimated noise floor)
    # Noise floor estimated from lowest 10% energy frames
    if num_frames >= 5:
        sorted_energies = np.sort(f_rms)
        noise_floor = float(np.mean(sorted_energies[:max(1, int(num_frames * 0.15))]))
        noise_floor = max(noise_floor, 1e-6)
        signal_energy = max(rms_val, 1e-6)
        snr_est = float(20.0 * np.log10(signal_energy / noise_floor))
        snr_est = max(0.0, min(60.0, snr_est))
    else:
        snr_est = 20.0 if rms_val > 0.01 else 5.0

    # Calculate Deductions
    score = 100
    reasons: List[str] = []

    # Clipping penalty
    if clipping_ratio > 0.10:
        score -= 40
        reasons.append(f"Severe clipping detected ({round(clipping_ratio * 100, 1)}% of samples)")
    elif clipping_ratio > 0.02:
        score -= 15
        reasons.append(f"Minor clipping detected ({round(clipping_ratio * 100, 1)}% of samples)")

    # Energy / Silence penalty
    if rms_val < 0.003:
        score -= 50
        reasons.append(f"Very low signal amplitude (RMS {round(rms_val, 5)})")
    elif rms_val < 0.01:
        score -= 20
        reasons.append(f"Low signal energy (RMS {round(rms_val, 4)})")

    if silence_ratio > 0.70:
        score -= 25
        reasons.append(f"High silence ratio ({round(silence_ratio * 100, 1)}%)")

    # Short duration penalty (< 0.25s)
    if duration_sec < 0.25:
        score -= 20
        reasons.append(f"Short audio window ({round(duration_sec, 2)}s)")

    # SNR penalty
    if snr_est < 6.0 and rms_val >= 0.003:
        score -= 20
        reasons.append(f"Low SNR estimation ({round(snr_est, 1)} dB)")

    score = max(0, min(100, score))

    if score >= 75:
        tier = "EXCELLENT"
    elif score >= 55:
        tier = "GOOD"
    elif score >= 40:
        tier = "FAIR"
    else:
        tier = "POOR"

    is_sufficient = score >= 40 and rms_val >= 0.003

    if len(reasons) == 0:
        reasons.append("Clean signal with sufficient vocal dynamic range")

    return {
        "quality_score": int(score),
        "quality_tier": tier,
        "is_sufficient": bool(is_sufficient),
        "reasons": reasons,
        "metrics": {
            "duration_seconds": round(duration_sec, 3),
            "rms_energy": round(rms_val, 5),
            "peak_amplitude": round(peak_val, 5),
            "clipping_ratio": round(clipping_ratio, 4),
            "estimated_snr_db": round(snr_est, 1),
            "silence_ratio": round(silence_ratio, 3),
        },
    }
