"""Phase 11: Evaluation script for VocalGuard AI V2.

Computes:
- Accuracy, Precision, Recall, F1
- False Positive Rate (FPR), False Negative Rate (FNR)
- Confusion Matrix

NON-NEGOTIABLE HONESTY NOTICE:
A small smoke-test collection MUST NOT be described as a production validation benchmark.
Representative benchmark scores require speaker-disjoint cross-evaluation on standard
corpuses (e.g. ASVspoof 2019/2021, In-The-Wild, WaveFake) with unseen speakers, diverse accents,
transcoded telephone codecs (G.711u/a, AMR, Opus), and acoustic room reverberations.
"""

import sys
import json
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.preprocessing import load_audio_bytes
from app.detector import VoiceDetector
from app.scoring import calculate_risk

BASE_DIR = Path(__file__).resolve().parent.parent
META_PATH = BASE_DIR / "audio_samples" / "metadata.json"

def run_evaluation(meta_path: Path = META_PATH):
    print("=" * 75)
    print("VOCALGUARD AI V2 — MODEL EVALUATION HARNESS")
    print("=" * 75)

    if not meta_path.exists():
        print(f"Error: Metadata file not found at {meta_path}")
        print("Run 'python scripts/generate_sample_audio.py' first.")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} audio sample records from {meta_path.name}")
    print("Initializing VoiceDetector...")
    detector = VoiceDetector()

    y_true = []  # 1 for fake, 0 for real
    y_pred = []
    y_scores = []
    results = []

    for item in samples:
        rel_path = item["file"]
        full_path = BASE_DIR / rel_path
        if not full_path.exists():
            print(f"Warning: File {full_path} missing; skipping.")
            continue

        with open(full_path, "rb") as af:
            audio_bytes = af.read()

        audio, sr = load_audio_bytes(audio_bytes)
        pred = detector.predict(audio, sr)
        risk = calculate_risk(pred["synthetic_score"])

        true_label = item["label"].lower()
        true_binary = 1 if true_label in ["fake", "synthetic", "spoof"] else 0
        pred_binary = 1 if pred["synthetic_score"] >= 0.50 else 0

        y_true.append(true_binary)
        y_pred.append(pred_binary)
        y_scores.append(pred["synthetic_score"])

        results.append({
            "file": item["file"],
            "speaker_id": item.get("speaker_id", "N/A"),
            "true_label": true_label,
            "pred_synthetic_score": round(pred["synthetic_score"], 4),
            "risk_score": risk["risk_score"],
            "verdict": risk["verdict"],
            "correct": (true_binary == pred_binary)
        })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print("-" * 75)
    print("DETAILED SAMPLE INFERENCE:")
    for r in results:
        status_sym = "[OK]" if r["correct"] else "[FAIL]"
        print(f"  {status_sym} {Path(r['file']).name:<28} | True: {r['true_label']:<5} | Score: {r['pred_synthetic_score']:>6.3f} | {r['verdict']}")

    print("-" * 75)
    print("CONFUSION MATRIX:")
    print(f"  True Positives  (Synthetic correctly caught):  {tp}")
    print(f"  True Negatives  (Genuine correctly passed):    {tn}")
    print(f"  False Positives (Genuine falsely flagged):     {fp}")
    print(f"  False Negatives (Synthetic missed):            {fn}")
    print("-" * 75)
    print("METRICS ON CURRENT SAMPLE COLLECTION:")
    print(f"  Accuracy:  {accuracy * 100:.1f}%")
    print(f"  Precision: {precision * 100:.1f}%")
    print(f"  Recall:    {recall * 100:.1f}%")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  FPR:       {fpr * 100:.1f}%")
    print(f"  FNR:       {fnr * 100:.1f}%")
    print("-" * 75)
    print("CRITICAL STATISTICAL DISCLAIMER:")
    print("These numbers are computed exclusively on the local smoke-test fixture set (N={total}).")
    print("They MUST NOT be cited as real-world operational accuracy or generalized performance.")
    print("=" * 75)

if __name__ == "__main__":
    run_evaluation()
