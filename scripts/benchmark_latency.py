"""Phase 10: End-to-end Latency Benchmark script for VocalGuard AI V2.

Measures separately:
- Preprocessing latency
- Feature extraction latency (Spectral, LFCC, Phase)
- Neural inference latency (Wav2Vec2 PyTorch)
- Total analysis latency

Reports mean, p50, p95, p99, min, max across configurable duration windows.
RULE: Never claim sub-200 ms CPU latency unless measured end-to-end under defined conditions.
"""

import sys
import time
import io
from pathlib import Path
import numpy as np
import soundfile as sf

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.preprocessing import load_audio_bytes
from app.detector import VoiceDetector
from app.features import extract_all_features
from app.scoring import calculate_risk

def create_synthetic_wav(duration_s: float, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # 440Hz + harmonic overtone
    samples = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def run_benchmark(durations=[0.25, 0.5, 1.0, 3.0], iterations=5):
    print("=" * 75)
    print("VOCALGUARD AI V2 — INFERENCE & PIPELINE LATENCY BENCHMARK")
    print("=" * 75)
    
    print("Initializing detector model...")
    detector = VoiceDetector()
    print(f"Model:  {detector.model_name}")
    print(f"Device: {detector.device}")
    print("-" * 75)
    
    # Warmup pass
    warmup_wav = create_synthetic_wav(1.0)
    audio_w, sr_w = load_audio_bytes(warmup_wav)
    _ = detector.predict(audio_w, sr_w)
    _ = extract_all_features(audio_w, sr_w)
    
    results = {}
    
    for dur in durations:
        print(f"\nBenchmarking Duration Window: {dur}s ({int(dur*1000)}ms) over {iterations} runs...")
        wav_bytes = create_synthetic_wav(dur)
        
        pre_times = []
        inf_times = []
        feat_times = []
        total_times = []
        
        for _ in range(iterations):
            # Preprocessing
            t0 = time.perf_counter()
            audio, sr = load_audio_bytes(wav_bytes)
            t_pre = (time.perf_counter() - t0) * 1000.0
            
            # Model forward pass
            t1 = time.perf_counter()
            m_res = detector.predict(audio, sr)
            t_inf = (time.perf_counter() - t1) * 1000.0
            
            # Feature extraction
            t2 = time.perf_counter()
            feats = extract_all_features(audio, sr)
            t_feat = (time.perf_counter() - t2) * 1000.0
            
            # Risk calculation
            _ = calculate_risk(m_res["synthetic_score"])
            t_total = (time.perf_counter() - t0) * 1000.0
            
            pre_times.append(t_pre)
            inf_times.append(t_inf)
            feat_times.append(t_feat)
            total_times.append(t_total)
            
        def stats(arr):
            return {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
            
        dur_stats = {
            "preprocessing": stats(pre_times),
            "inference": stats(inf_times),
            "feature_extraction": stats(feat_times),
            "total_analysis": stats(total_times),
        }
        results[f"{dur}s"] = dur_stats
        
        print(f"  [Preprocessing]    Mean: {dur_stats['preprocessing']['mean']:.2f}ms | p50: {dur_stats['preprocessing']['p50']:.2f}ms | p95: {dur_stats['preprocessing']['p95']:.2f}ms")
        print(f"  [Neural Inference] Mean: {dur_stats['inference']['mean']:.2f}ms | p50: {dur_stats['inference']['p50']:.2f}ms | p95: {dur_stats['inference']['p95']:.2f}ms")
        print(f"  [Features (LFCC)]  Mean: {dur_stats['feature_extraction']['mean']:.2f}ms | p50: {dur_stats['feature_extraction']['p50']:.2f}ms | p95: {dur_stats['feature_extraction']['p95']:.2f}ms")
        print(f"  --> TOTAL PIPELINE: Mean: {dur_stats['total_analysis']['mean']:.2f}ms | p50: {dur_stats['total_analysis']['p50']:.2f}ms | p95: {dur_stats['total_analysis']['p95']:.2f}ms")

    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY & HONESTY VERIFICATION:")
    for dur, s in results.items():
        sub200 = s['total_analysis']['p95'] < 200.0
        status = "SUB-200ms ACHIEVED" if sub200 else "EXCEEDS 200ms THRESHOLD"
        print(f"  Window {dur:>5}: p50={s['total_analysis']['p50']:6.1f}ms | p95={s['total_analysis']['p95']:6.1f}ms --> [{status}]")
    print("=" * 75)
    return results

if __name__ == "__main__":
    run_benchmark(durations=[0.25, 0.5, 1.0, 3.0], iterations=5)
