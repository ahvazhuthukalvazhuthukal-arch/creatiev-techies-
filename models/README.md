# VocalGuard AI V2 — Model Documentation

## Active Prototype Model

- **Model Identifier**: `garystafford/wav2vec2-deepfake-voice-detector`
- **Architecture**: Fine-tuned Wav2Vec 2.0 Base (`Wav2Vec2ForSequenceClassification`)
- **Sampling Rate Requirement**: 16,000 Hz (16 kHz mono)
- **Input Representation**: Raw 1D audio waveform samples (normalized float32)
- **Label Mapping (`id2label`)**:
  - `0`: `real` (Genuine human voice)
  - `1`: `fake` (Synthetic / cloned / vocoded audio)
- **Label Verification**:
  - Verified programmatically via `scripts/inspect_model_labels.py`.
  - The detector inspects the model's configuration dictionary using canonical candidate tokens: `{"fake", "synthetic", "spoof", "deepfake", "generated"}`.
  - If ambiguous or missing, the system aborts with an explicit error rather than guessing.

## Research Target Architecture (SIH Roadmap)

The PPT target architecture defines a specialized compact pipeline to replace bulky transformer backbones for ultra-low latency telecom environments:
1. **Frontend Feature Extractor**:
   - Linear Frequency Cepstral Coefficients (LFCC) — 20 to 60 filterbanks with delta and double-delta coefficients.
   - Complex STFT Phase Spectrum Trajectories & Group Delay.
2. **Backbone Classifiers**:
   - **LightCNN-29**: Max-Feature-Map (MFM) activation for artifact edge detection.
   - **ResNet-18 / ResNet-34**: 2D convolutional residual network trained on spectrogram/LFCC representations.
3. **Execution Runtime**:
   - Exported to ONNX and quantized to INT8 with dynamic range quantization.
   - Targeted sub-50ms inference on commodity x86/ARM CPU servers.
