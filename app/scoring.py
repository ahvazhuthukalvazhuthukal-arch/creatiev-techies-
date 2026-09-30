"""Risk scoring module for VocalGuard AI V2.

NON-NEGOTIABLE HONESTY NOTICE:
The score produced here combines the neural model's synthetic-class softmax probability
with acoustic forensic anomaly telemetry (LFCC vocoder artifacts, spectral flatness,
centroid displacement, and biological vocal fold micro-jitter).
It is a heuristic classification and forensic anomaly score, NOT a calibrated real-world
probability of criminal fraud.

Thresholds:
  < 30.00:  LOW RISK
  30.00 – 49.99:  MODERATE / INCONCLUSIVE
  50.00 – 74.99:  SUSPICIOUS
  >= 75.00: CRITICAL RISK
"""

from typing import Optional, Dict, Any
import numpy as np
from .config import THRESHOLD_CRITICAL, THRESHOLD_SUSPICIOUS, THRESHOLD_MODERATE

def compute_forensic_anomaly_score(features: Dict[str, Any]) -> float:
    """Computes an acoustic vocoder artifact anomaly score (0.0 to 100.0) based on
    LFCC, spectral flatness, centroid, and biological pitch dynamics.
    """
    flat = features.get("spectral_flatness_mean", 0.0)
    cent = features.get("spectral_centroid_mean_hz", 500.0)
    lfcc_std = features.get("lfcc_std", 14.0)
    jitter = features.get("pitch_jitter", 0.008)

    # 1. Spectral Flatness: vocoders introduce white-noise dispersion
    # Natural voice: < 1e-5 (log10 < -5.0). Vocoder: > 1e-3 (log10 > -3.0)
    flat_log = np.log10(max(1e-9, flat))
    flat_score = min(30.0, max(0.0, (flat_log + 5.5) * 12.0))

    # 2. Spectral Centroid: vocoder high-frequency concentration (> 750 Hz)
    cent_score = min(25.0, max(0.0, (cent - 750.0) / 50.0))

    # 3. LFCC Cepstral Dispersion: natural speech has high variance (> 11.5); vocoders have compressed linear cepstra (< 7.0)
    lfcc_score = min(25.0, max(0.0, (11.5 - lfcc_std) * 3.5)) if lfcc_std < 11.5 else 0.0

    # 4. Biological Pitch Jitter & Vocal Dynamics
    # Natural human speech has jitter ~0.004 to 0.02. Robotic TTS pitch has jitter < 0.003
    jitter_score = min(20.0, max(0.0, (0.003 - jitter) * 10000.0)) if (jitter > 0 and jitter < 0.003) else 0.0

    return float(np.clip(flat_score + cent_score + lfcc_score + jitter_score, 0.0, 100.0))

def calculate_risk(synthetic_score: float, features: Optional[Dict[str, Any]] = None) -> dict:
    model_risk = float(synthetic_score) * 100.0
    
    if features is not None:
        anomaly_score = compute_forensic_anomaly_score(features)
        # Multi-signal fusion: combines neural classifier with acoustic vocoder telemetry
        risk = max(model_risk, 0.40 * model_risk + 0.60 * anomaly_score)
        score_type = "forensic acoustic ensemble score (neural + LFCC/spectral anomaly); not calibrated fraud probability"
    else:
        anomaly_score = 0.0
        risk = model_risk
        score_type = "synthetic-class model score; not calibrated fraud probability"

    risk = max(0.0, min(100.0, risk))

    if risk >= THRESHOLD_CRITICAL:
        verdict = "CRITICAL RISK"
    elif risk >= THRESHOLD_SUSPICIOUS:
        verdict = "SUSPICIOUS"
    elif risk >= THRESHOLD_MODERATE:
        verdict = "MODERATE / INCONCLUSIVE"
    else:
        verdict = "LOW RISK"

    return {
        "risk_score": round(risk, 2),
        "model_score": round(model_risk, 2),
        "anomaly_score": round(anomaly_score, 2),
        "verdict": verdict,
        "score_type": score_type
    }
