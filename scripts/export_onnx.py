"""Phase 14 & 15: Export Wav2Vec2 to ONNX and Validate Numerical Parity.

Steps:
1. Export PyTorch AutoModelForAudioClassification to ONNX format with dynamic batch & sequence axes.
2. Load ONNX graph using onnxruntime.InferenceSession.
3. Compare PyTorch vs ONNX Runtime logits on identical input audio.
4. Measure maximum absolute drift: |PyTorch - ONNX| < 1e-4.
5. Report model file size and runtime comparison.
"""

import sys
import io
import time
import os
from pathlib import Path

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detector import VoiceDetector
from app.config import BASE_DIR

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ONNX_PATH = MODELS_DIR / "wav2vec2_deepfake.onnx"

def export_and_validate():
    print("=" * 75)
    print("VOCALGUARD AI V2 — ONNX EXPORT & NUMERICAL PARITY VALIDATION")
    print("=" * 75)
    
    print("Loading PyTorch detector...")
    detector = VoiceDetector()
    model = detector.model
    model.eval()
    
    # 1 second dummy input: [batch_size, sequence_length]
    dummy_input = torch.zeros((1, 16000), dtype=torch.float32)
    
    print(f"Exporting model to {ONNX_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    file_size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK] ONNX export complete. File size: {file_size_mb:.2f} MB")
    
    # 2. Validate with ONNX Runtime
    print("\nInitializing ONNX Runtime session...")
    ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    
    # Test on sine wave
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    test_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    # PyTorch forward pass
    pt_inputs = detector.extractor(test_audio, sampling_rate=16000, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_outputs = model(**pt_inputs)
        pt_logits = pt_outputs.logits.numpy()
    pt_time = (time.perf_counter() - t0) * 1000.0
    
    # ONNX Runtime forward pass
    ort_inputs = {"input_values": pt_inputs["input_values"].numpy()}
    t1 = time.perf_counter()
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_logits = ort_outputs[0]
    ort_time = (time.perf_counter() - t1) * 1000.0
    
    # Parity check
    max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    print("-" * 75)
    print(f"PyTorch Logits:       {pt_logits}")
    print(f"ONNX Runtime Logits:  {ort_logits}")
    print(f"Max Absolute Drift:   {max_diff:.6e}")
    print(f"PyTorch Latency:      {pt_time:.2f} ms")
    print(f"ONNX Runtime Latency: {ort_time:.2f} ms")
    print("-" * 75)
    
    if max_diff < 1e-4:
        print("[VERIFIED] Numerical parity confirmed within strict tolerance (< 1e-4).")
    else:
        print("[WARNING] Numerical drift exceeds standard tolerance.")
        
    print("=" * 75)

if __name__ == "__main__":
    export_and_validate()
