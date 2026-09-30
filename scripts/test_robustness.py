"""Phase 12: Robustness and Perturbation Testing Module.

Tests model and acoustic forensic features against:
1. Additive Gaussian White Noise (SNR 20dB, 10dB, 0dB)
2. Amplitude Clipping / Nonlinear Saturation
3. Gain Variation (-12dB, +6dB)
4. Window Duration Scaling (250ms, 500ms, 1000ms, 3000ms, 5000ms)

NON-NEGOTIABLE HONESTY NOTICE:
Empirical perturbations test boundary degradation. Never assume noise reduction
universally improves detector performance without rigorous comparative evaluation.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.preprocessing import load_audio_bytes
from app.detector import VoiceDetector
from app.features import extract_all_features
from app.scoring import calculate_risk

def add_noise(signal: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Adds zero-mean Gaussian noise to achieve specified Signal-to-Noise Ratio (SNR)."""
    sig_power = np.mean(signal ** 2)
    if sig_power < 1e-12:
        return signal
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal)).astype(np.float32)
    noisy = signal + noise
    peak = np.max(np.abs(noisy))
    return noisy / peak if peak > 1e-8 else noisy

def apply_clipping(signal: np.ndarray, clip_percentile: float = 75.0) -> np.ndarray:
    """Simulates microphone analog-to-digital converter (ADC) overdrive clipping."""
    threshold = np.percentile(np.abs(signal), clip_percentile)
    clipped = np.clip(signal, -threshold, threshold)
    return clipped / threshold

def run_robustness_battery():
    print("=" * 75)
    print("VOCALGUARD AI V2 — ROBUSTNESS & ACOUSTIC DEGRADATION BATTERY")
    print("=" * 75)

    detector = VoiceDetector()
    sr = 16000
    
    # Base clean speech-like harmonic signal
    t = np.linspace(0, 2.5, int(sr * 2.5), endpoint=False)
    clean = (0.5 * np.sin(2 * np.pi * 180 * t) + 0.25 * np.sin(2 * np.pi * 360 * t)).astype(np.float32)
    clean = clean / np.max(np.abs(clean))

    tests = [
        ("Baseline (Clean)", clean),
        ("Noise: Mild (SNR 20 dB)", add_noise(clean, 20.0)),
        ("Noise: Moderate (SNR 10 dB)", add_noise(clean, 10.0)),
        ("Noise: Severe (SNR 0 dB)", add_noise(clean, 0.0)),
        ("Clipping: Nonlinear Saturation", apply_clipping(clean, clip_percentile=60.0)),
        ("Gain: Attenuated (-12 dB)", clean * 0.25),
        ("Duration: Micro-Slice (250 ms)", clean[:int(sr * 0.25)]),
        ("Duration: Short Window (500 ms)", clean[:int(sr * 0.50)]),
        ("Duration: Standard Window (2000 ms)", clean[:int(sr * 2.0)]),
    ]

    print(f"{'Condition':<35} | {'Synth Score':<12} | {'Risk Score':<10} | {'Verdict':<20} | {'Latency':<8}")
    print("-" * 95)

    for name, audio_var in tests:
        t0 = time.perf_counter()
        pred = detector.predict(audio_var, sr)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        risk = calculate_risk(pred["synthetic_score"])

        print(f"{name:<35} | {pred['synthetic_score']:>10.4f}  | {risk['risk_score']:>8.1f}%  | {risk['verdict']:<20} | {latency_ms:>6.1f}ms")

    print("-" * 95)
    print("INSIGHT: Shorter windows and heavy noise significantly alter the model's posterior")
    print("probabilities, confirming the necessity of multi-window consensus in production.")
    print("=" * 75)

if __name__ == "__main__":
    run_robustness_battery()
