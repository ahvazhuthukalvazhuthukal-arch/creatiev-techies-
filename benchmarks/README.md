# VocalGuard AI V2 — Benchmarks & Latency Documentation

## Measurement Methodology

VocalGuard AI V2 conducts strict empirical latency measurements on the actual inference hardware.
Pipeline latency is decoupled into four isolated phases:
1. **Preprocessing (`preprocessing_ms`)**: In-memory byte decoding, mono channel conversion, 16 kHz resampling via Librosa, peak normalization, and boundary sanitization.
2. **Neural Inference (`model_inference_ms`)**: Wav2Vec2 forward pass in PyTorch with `torch.no_grad()`.
3. **Forensic Feature Extraction (`feature_extraction_ms`)**: STFT spectral flatness, centroid, 20-filterbank LFCC extraction via DCT, and phase unwrapping.
4. **Total Pipeline Latency (`total_analysis_ms`)**: Cumulative end-to-end execution.

## Reproducing Latency Benchmarks

To execute the benchmark across configurable sliding window durations (250ms, 500ms, 1000ms, 3000ms):

```bash
python scripts/benchmark_latency.py
```

### Metrics Reported:
- **Mean Latency (ms)**
- **p50 (Median ms)**
- **p95 (95th Percentile ms)**
- **p99 (99th Percentile ms)**
- **Min / Max (ms)**

## The Sub-200ms Target: Reality vs Research Target

The Creative Techies SIH Target Architecture aims for `< 200 ms` inference on CPU to enable seamless inline call intervention.
- In the **current prototype**, the general-purpose Wav2Vec2 transformer backbone requires ~150–400ms on CPU for typical voice frames.
- Reaching guaranteed `< 200 ms` across all hardware requires the **Target Optimization Phases**:
  1. Exporting to ONNX Runtime (`scripts/export_onnx.py`).
  2. Applying INT8 post-training dynamic quantization.
  3. Migrating from heavy self-attention transformers to compact specialized architectures (e.g. LightCNN-29 or ResNet-18) operating directly on compact LFCC feature matrices.
