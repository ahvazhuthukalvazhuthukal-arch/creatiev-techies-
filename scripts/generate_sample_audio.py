"""Generates local test audio fixtures for evaluation smoke tests.
Generates genuine-like voice harmonic fixtures and vocoder/cloning-like synthetic fixtures.
Outputs:
  - audio_samples/real/sample_real_01.wav
  - audio_samples/real/sample_real_02.wav
  - audio_samples/synthetic/sample_synthetic_01.wav
  - audio_samples/synthetic/sample_synthetic_02.wav
  - audio_samples/metadata.json
"""

import json
from pathlib import Path
import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
REAL_DIR = BASE_DIR / "audio_samples" / "real"
SYN_DIR = BASE_DIR / "audio_samples" / "synthetic"
META_PATH = BASE_DIR / "audio_samples" / "metadata.json"

REAL_DIR.mkdir(parents=True, exist_ok=True)
SYN_DIR.mkdir(parents=True, exist_ok=True)

def generate_natural_harmonic_tone(duration=2.5, sr=16000, f0=180.0):
    """Simulates vocal fold harmonic vibration with natural micro-jitter/tremor."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Pitch jitter
    jitter = 0.02 * np.sin(2 * np.pi * 5.0 * t)
    f_t = f0 * (1.0 + jitter)
    phase = 2 * np.pi * np.cumsum(f_t) / sr
    
    # Harmonics (1st, 2nd, 3rd, 4th formants)
    signal = (
        0.5 * np.sin(phase) +
        0.25 * np.sin(2 * phase) +
        0.15 * np.sin(3 * phase) +
        0.08 * np.sin(4 * phase)
    )
    # Natural amplitude envelope (vocal breathing attack and release)
    env = np.sin(np.pi * t / duration) ** 0.5
    signal = signal * env
    peak = np.max(np.abs(signal))
    return (signal / peak).astype(np.float32)

def generate_vocoder_synthetic_artifact(duration=2.5, sr=16000, f0=180.0):
    """Simulates neural vocoder artifacts: high-frequency buzzy harmonics & phase discontinuities."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fixed mechanical pitch (zero jitter)
    phase = 2 * np.pi * f0 * t
    # Overly sharp rectangular/pulse wave (vocoder buzz)
    signal = np.sign(np.sin(phase)) * 0.4
    # Discontinuous phase injection / high frequency chirp
    hf_chirp = 0.15 * np.sin(2 * np.pi * 3200 * t)
    signal += hf_chirp
    # Mechanical linear envelope
    signal = signal * np.clip(t / 0.1, 0, 1) * np.clip((duration - t) / 0.1, 0, 1)
    peak = np.max(np.abs(signal))
    return (signal / peak).astype(np.float32)

def main():
    print("Generating local smoke-test audio fixtures...")
    sr = 16000
    
    # Real 01: Male fundamental simulation
    r1 = generate_natural_harmonic_tone(duration=2.5, sr=sr, f0=130.0)
    sf.write(str(REAL_DIR / "sample_real_01.wav"), r1, sr, subtype="PCM_16")

    # Real 02: Female fundamental simulation
    r2 = generate_natural_harmonic_tone(duration=3.0, sr=sr, f0=220.0)
    sf.write(str(REAL_DIR / "sample_real_02.wav"), r2, sr, subtype="PCM_16")

    # Synthetic 01: Robotized buzz artifact
    s1 = generate_vocoder_synthetic_artifact(duration=2.5, sr=sr, f0=130.0)
    sf.write(str(SYN_DIR / "sample_synthetic_01.wav"), s1, sr, subtype="PCM_16")

    # Synthetic 02: High-freq vocoder artifact
    s2 = generate_vocoder_synthetic_artifact(duration=3.0, sr=sr, f0=220.0)
    sf.write(str(SYN_DIR / "sample_synthetic_02.wav"), s2, sr, subtype="PCM_16")

    metadata = [
        {
            "file": "audio_samples/real/sample_real_01.wav",
            "label": "real",
            "speaker_id": "SPK_GEN_01",
            "language": "en",
            "accent": "General Indian",
            "source_type": "harmonic_simulation",
            "synthesis_system": "none",
            "device": "simulated_mic",
            "codec": "PCM_16",
            "noise_condition": "clean"
        },
        {
            "file": "audio_samples/real/sample_real_02.wav",
            "label": "real",
            "speaker_id": "SPK_GEN_02",
            "language": "en",
            "accent": "Standard American",
            "source_type": "harmonic_simulation",
            "synthesis_system": "none",
            "device": "simulated_mic",
            "codec": "PCM_16",
            "noise_condition": "clean"
        },
        {
            "file": "audio_samples/synthetic/sample_synthetic_01.wav",
            "label": "fake",
            "speaker_id": "SPK_SYN_01",
            "language": "en",
            "accent": "Synthetic Neutral",
            "source_type": "vocoder_simulation",
            "synthesis_system": "vocoder_buzz_gen",
            "device": "direct_render",
            "codec": "PCM_16",
            "noise_condition": "clean"
        },
        {
            "file": "audio_samples/synthetic/sample_synthetic_02.wav",
            "label": "fake",
            "speaker_id": "SPK_SYN_02",
            "language": "en",
            "accent": "Synthetic Neutral",
            "source_type": "vocoder_simulation",
            "synthesis_system": "vocoder_buzz_gen",
            "device": "direct_render",
            "codec": "PCM_16",
            "noise_condition": "clean"
        }
    ]

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Generated 2 real and 2 synthetic audio fixtures.")
    print(f"[OK] Metadata catalog written to {META_PATH}")

if __name__ == "__main__":
    main()
