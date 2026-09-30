# VOCALGUARD AI V2 — LIVE MODE AUDIT & GAP ANALYSIS
**Project**: VocalGuard AI V2 | **Team**: Creative Techies | **Track**: SIH26104

---

## 1. Executive Summary

This document provides a factual, code-grounded audit of the real-time detection capabilities in the VocalGuard AI V2 repository prior to the live-pipeline overhaul. It answers the 19 forensic audit questions (A through S) and maps what is active, what is experimental, what is missing, and what belongs to future roadmap targets.

---

## 2. Answers to Forensic Audit Questions

### A. What happens when Live Mode starts?
The browser requests microphone access via `navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000, echoCancellation: false, noiseSuppression: false, autoGainControl: false } })`. An `AudioContext` and a `ScriptProcessorNode` (buffer size 4096) are created. A WebSocket connection is opened to `ws://<host>:<port>/ws/live`. The UI initializes a VU meter, sets status to `LISTENING / STREAMING`, and streams audio buffers to the server.

### B. Where does microphone audio go?
Microphone audio captured in the browser `onaudioprocess` callback is downsampled in JavaScript to 16,000 Hz, converted to 16-bit signed PCM integers (`Int16Array`), and transmitted as raw binary frames over the WebSocket to `/ws/live`. On the server, chunks are appended to a bounded in-memory float32 NumPy buffer (`audio_buffer`).

### C. What size are incoming chunks?
The browser uses a `ScriptProcessorNode` buffer size of 4096 samples. At the hardware rate (typically 44.1kHz or 48kHz on Windows), downsampling produces ~1365 samples (~85 ms of audio at 16kHz) per binary WebSocket frame (2730 bytes).

### D. Are chunks accumulated?
Yes. On the server side, chunks are concatenated into a bounded rolling NumPy array (`audio_buffer = np.concatenate([audio_buffer, samples])`).

### E. Is the AI detector called during Live Mode?
Yes. When the rolling buffer contains at least `current_window_samples` (default 500 ms = 8,000 samples) and the inference throttle interval (0.4 s) has elapsed, a slice of the latest audio is dispatched to a background thread executor via `loop.run_in_executor`.

### F. How often is inference performed?
Inference is executed at most once every 400 ms (`now - last_eval_time >= 0.4`), provided the previous inference has completed (`is_evaluating == False`). If inference takes 250 ms, the effective cadence is approximately 2 to 2.5 inferences per second.

### G. Is the audio converted to 16 kHz?
Yes. The client implements an offline accumulator downsampler (`downsampleTo16k`) that converts whatever hardware sample rate the browser enforces (44.1 kHz, 48 kHz, or 96 kHz) to 16,000 Hz before transmission. On the server, `TARGET_SR = 16000` is asserted.

### H. Is stereo converted to mono?
Yes. Client-side `getUserMedia` requests `channelCount: 1`. In JavaScript, `e.inputBuffer.getChannelData(0)` extracts mono channel 0. The server-side preprocessing module also implements `np.mean(audio, axis=1)` if multi-channel arrays are encountered.

### I. Is there a rolling buffer?
Yes. The server maintains `audio_buffer` bounded to `STREAMING_BUFFER_MAX_SECONDS * TARGET_SR` (5 seconds = 80,000 float32 samples). Older samples are trimmed from the head (`audio_buffer = audio_buffer[-max_buffer_samples:]`).

### J. Is there a quality gate?
Partially. The server previously checked only for silence (`rms < 0.003`) and peak normalization. There was **no comprehensive audio quality score**, no SNR estimation, no clipping ratio calculation, and no automatic transition to an `INCONCLUSIVE` state for degraded audio.

### K. Is voice activity detection implemented?
Only a primitive RMS energy check (`rms < 0.003`). There is no multi-feature VAD (energy + zero-crossing rate + spectral flux) and no `speech_ratio` calculation across the rolling window.

### L. Are LFCC features actually calculated?
Yes. `app/features.py` extracts 20-filterbank Linear Frequency Cepstral Coefficients via `librosa.feature.mfcc(..., mel_kws={'htk': False})` using linear triangular filterbanks, returning `lfcc_mean`, `lfcc_std`, and `lfcc_abs_mean`. **Crucial honesty disclaimer**: These are acoustic summary features, NOT a trained LFCC classifier.

### M. Are phase features actually calculated?
Yes. `app/features.py` computes Short-Time Fourier Transform (STFT) phase angles via `np.angle(stft)` and extracts `phase_mean`, `phase_std`, and second-order phase derivative variance `phase_diff_std`. They serve as vocoder-artifact indicators, not a standalone classifier.

### N. Is the neural model actually called?
Yes. `garystafford/wav2vec2-deepfake-voice-detector` is loaded as a singleton on CPU and evaluated on the normalized audio window using PyTorch `torch.no_grad()`.

### O. Is the risk score updated continuously?
Yes, every inference result triggers a WebSocket JSON payload with `risk.risk_score`, `risk.risk_level`, and `risk.verdict`.

### P. Is temporal smoothing implemented?
**No, previously missing.** Each incoming window was evaluated in isolation. A single noisy or unvoiced window could cause the risk score to spike to 85% and immediately drop to 15% on the next window. There was no rolling history, no exponential moving average, no median filtering, and no hysteresis state machine.

### Q. Is the frontend displaying historical scores?
**No, previously missing.** The frontend displayed a snapshot gauge, numerical labels, and instantaneous values, but lacked a rolling time-series graph (0–60s) showing how synthetic score evolves over time alongside decision thresholds.

### R. Is audio being stored?
**No.** Audio processing is strictly in-memory (volatile RAM). Slices are evaluated and discarded; no `.wav` files, temporary buffers, or logs are written to disk.

### S. What is the actual latency?
Empirical measurements on CPU:
- 250 ms window: ~153 ms p50
- 500 ms window: ~170 ms p50
- 1000 ms window: ~265 ms p50
- 3000 ms window: ~701 ms p50
*(Feature extraction adds ~8–15 ms, Preprocessing adds ~1–3 ms).*

---

## 3. Component Status Matrix

### CURRENTLY IMPLEMENTED & ACTIVE
1. **FastAPI Server** (`app/main.py`) with HTTP endpoints (`/health`, `/analyze`, `/live-demo`) and WebSocket `/ws/live`.
2. **Pretrained Wav2Vec2 Classifier** (`app/detector.py`) with explicit `id2label` mapping ({0: 'real', 1: 'fake'}).
3. **In-Memory Zero-Storage Preprocessing** (`app/preprocessing.py`) enforcing 16kHz mono, float32, peak normalization, NaN/Inf sanitization.
4. **Forensic Feature Extraction** (`app/features.py`):
   - Spectral Flatness mean & std
   - Spectral Centroid mean & std (Hz)
   - 20-filterbank LFCC mean, std, abs_mean
   - STFT Phase mean, std, diff_std
   - Fundamental frequency pitch jitter
5. **Multi-Signal Risk Scoring** (`app/scoring.py`): Weighted risk combining neural softmax score with forensic acoustic anomaly indicators.
6. **Zero-Queue-Lag WebSocket Dispatcher**: Decouples audio reception from model inference via `is_evaluating` throttle and threadpool execution.
7. **Client-Side Hardware Rate Downsampler** in JavaScript to prevent the 3x time-dilation / pitch-drop bug on Windows.

### MISSING (TARGET OF THIS SPRINT)
1. **Audio Quality Gate Module** (`app/quality.py`): Multi-metric quality assessment (RMS, clipping ratio, SNR estimate, silence ratio) producing a 0–100 quality score and an `INCONCLUSIVE` verdict on poor audio.
2. **Temporal Evidence Aggregator** (`app/temporal.py`): Rolling buffer of past N window predictions, rolling mean, median, max, variance, and persistent elevated-window ratios.
3. **Hysteresis State Machine**: 5 distinct operational states (`INITIALIZING`, `ANALYZING`, `LOW_RISK`, `SUSPICIOUS`, `HIGH_RISK`, `INCONCLUSIVE`) requiring consecutive confirmations to transition into and out of high alert.
4. **Policy Arbitration Engine** (`app/policy.py`): Decouples forensic evidence from operational enforcement (`ALLOW`, `MONITOR`, `WARN_VERIFY`, `ESCALATE`, `BLOCK_SIMULATION`).
5. **Real-Time Rolling Canvas Graph**: Client-side SVG/Canvas chart tracking the last 30–60 seconds of synthetic scores against 50% and 75% thresholds.
6. **Live Forensic Event Log**: Timestamped forensic trace of speech detection, window evaluation, and alert state transitions.
7. **"Why This Result?" Forensic Breakdown Drawer**: Transparent, interactive breakdown of all calculated metrics and explicit forensic rationales.
8. **Automated Live Window Latency Benchmark**: Rigorous empirical measurement script across 250ms, 500ms, 750ms, 1000ms, and 3000ms windows.

### EXPERIMENTAL
1. **ONNX Export & Runtime**: `wav2vec2_deepfake.onnx` generated with parity drift < 5.58e-06.
2. **Acoustic Heuristic Fusion**: LFCC and phase indicators contribute to the risk score via empirical heuristic thresholds, not a trained classifier.
3. **SIP / Asterisk Call Intervention**: Simulated via `scripts/simulate_sip.py` mock bridge; no live PBX connected.

### TARGET ROADMAP (NOT YET IMPLEMENTED)
1. **Trained LightCNN-29 / ResNet-18** frontends on ASVspoof 2019/2021.
2. **Hardware-accelerated INT8 Quantization** for sub-100ms ARM/x86 inference.
3. **RNNoise / DeepFilternet** C++ DSP pre-filters for extreme noisy field conditions.
4. **Native Asterisk ARI / WebRTC SIP Trunk** inline media proxy.
