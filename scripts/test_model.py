"""Phase 3: Smoke test script for VoiceDetector.
Runs a forward pass on synthetic silent and sinusoidal test audio arrays.
NOTE: This is a software pipeline verification, not a quality or accuracy benchmark.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detector import VoiceDetector

def main():
    print("Initializing detector...")
    detector = VoiceDetector()
    
    # 2 seconds of 16kHz silence
    silence = np.zeros(32000, dtype=np.float32)
    print("\n--- Smoke Test 1: Silent Array (2.0s) ---")
    res_silence = detector.predict(silence, 16000)
    print("Class scores:", res_silence["class_scores"])
    print("Synthetic score:", res_silence["synthetic_score"])
    
    # 2 seconds of 440Hz sine wave
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    print("\n--- Smoke Test 2: Sine Wave 440Hz (2.0s) ---")
    res_sine = detector.predict(sine, 16000)
    print("Class scores:", res_sine["class_scores"])
    print("Synthetic score:", res_sine["synthetic_score"])
    print("\n[OK] VoiceDetector smoke test completed successfully.")
    print("Disclaimer: Model outputs on synthetic tones do not reflect real human/clone evaluation.")

if __name__ == "__main__":
    main()
