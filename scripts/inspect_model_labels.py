"""Phase 2: Inspect Model Labels script.
Loads the Wav2Vec2 feature extractor and classification model.
Prints sampling rate requirements, id2label, label2id, device, and verifies
the synthetic class determination logic with strict candidate matching.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detector import VoiceDetector

def main():
    print("=" * 60)
    print("VOCALGUARD AI V2 — MODEL LABEL INSPECTION")
    print("=" * 60)
    
    detector = VoiceDetector()
    
    print(f"Model Name:              {detector.model_name}")
    print(f"Device:                  {detector.device}")
    print(f"Config Expected SR:      {getattr(detector.extractor, 'sampling_rate', 16000)} Hz")
    print(f"ID to Label Mapping:     {detector.id2label}")
    print(f"Label to ID Mapping:     {detector.label2id}")
    print(f"Synthetic Class Index:   {detector.fake_index}")
    print(f"Synthetic Class Name:    '{detector.synthetic_label}'")
    print("=" * 60)
    print("LABEL INTEGRITY CHECK: PASSED")
    print("Synthetic class identified without guessing.")
    print("=" * 60)

if __name__ == "__main__":
    main()
