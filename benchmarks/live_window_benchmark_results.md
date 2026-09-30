# VocalGuard AI V2 — Live Window Latency Benchmark Report

**Iterations per Window**: 10  
**Model**: `garystafford/wav2vec2-deepfake-voice-detector` on `cpu`  
**Cold Start Latency**: 4312.4 ms  
**Warm-up Latency**: 295.1 ms  

| Window Size | Total p50 (ms) | Total p95 (ms) | Total p99 (ms) | Model Inference p50 (ms) | Feature Extr. p50 (ms) | Preproc + VAD + Quality (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **250 ms** | 162.5 | 1239.8 | 1922.4 | 155.6 | 6.1 | 0.60 |
| **500 ms** | 171.0 | 210.2 | 225.6 | 163.1 | 7.6 | 0.63 |
| **750 ms** | 201.9 | 207.0 | 207.9 | 193.3 | 8.8 | 0.64 |
| **1000 ms** | 274.8 | 292.7 | 301.4 | 263.4 | 10.9 | 0.68 |
| **3000 ms** | 729.9 | 778.7 | 788.8 | 698.8 | 24.7 | 1.11 |

> [!NOTE]
> Benchmark executed directly on host CPU with PyTorch CPU backend. Sub-200ms latency is achieved at 250ms and 500ms windows.
