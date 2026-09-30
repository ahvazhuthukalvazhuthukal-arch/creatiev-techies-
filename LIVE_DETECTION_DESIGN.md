# VOCALGUARD AI V2 — LIVE DETECTION PIPELINE DESIGN
**Project**: VocalGuard AI V2 | **Team**: Creative Techies | **Track**: SIH26104

---

## 1. Pipeline Architecture Flowchart

```text
[ Browser / Microphone ]
         │ (16kHz PCM downsampled, Mono, Int16)
         ▼
[ WebSocket Endpoint /ws/live ]
         │
         ▼
[ In-Memory Bounded Ring Buffer (Max 5.0s float32) ]
         │
   Cadence Check (Throttle >= 400ms & is_evaluating == False)
         │
         ├─── YES ───► Extract Latest Window (e.g. 500ms / 1000ms)
         │                    │
         │                    ▼
         │             [ Step 1: VAD & Speech Ratio Check ]
         │                    │
         │                    ├── Speech Ratio < 0.20 ──► Emit "SILENCE_GATE"
         │                    │                           (State: ANALYZING/IDLE)
         │                    └── Speech Ratio >= 0.20
         │                            │
         │                            ▼
         │             [ Step 2: Audio Quality Gate ]
         │                    │ (RMS, Clipping, SNR est., Silence ratio)
         │                    │
         │                    ├── Quality Score < 40 ──► Emit "INCONCLUSIVE"
         │                    │                          (Degraded Audio Flag)
         │                    └── Quality Score >= 40 (Valid Speech Window)
         │                            │
         │                            ▼ (Non-blocking Threadpool Executor)
         │             [ Step 3: Neural Model Inference ]
         │                    │ (Wav2Vec2 Deepfake Detector on CPU)
         │                    │  Returns: synthetic_score, class logits
         │                    │
         │                    ▼
         │             [ Step 4: Supporting Forensic Feature Extraction ]
         │                    │  Spectral Flatness & Centroid
         │                    │  LFCC (20-filterbank) Statistics
         │                    │  STFT Phase Dispersion
         │                    │  Pitch Jitter Dynamics
         │                    │
         │                    ▼
         │             [ Step 5: Multi-Signal Evidence Fusion ]
         │                    │ (Synthesizes Neural + Acoustic Indicators)
         │                    │
         │                    ▼
         │             [ Step 6: Temporal Evidence Aggregator ]
         │                    │  Maintains rolling queue of past N predictions
         │                    │  Computes rolling mean, median, max, variance
         │                    │  Hysteresis State Machine (Transitions 5 States)
         │                    │
         │                    ▼
         │             [ Step 7: Policy Arbitration Engine ]
         │                    │ (ALLOW, MONITOR, WARN_VERIFY, ESCALATE, BLOCK_SIMULATION)
         │                    │
         │                    ▼
         │             [ Step 8: Structured WebSocket Broadcast ]
         │                    │
         └─── NO ────► Accumulate audio chunks in ring buffer (Zero Lag)
                               │
                               ▼
            [ Live Cyber-Forensic Dashboard ]
            ├── Dynamic Rolling Timeline Graph (0–60s)
            ├── 5-State Status Badge + Alert Buzzer
            ├── Forensic Event Log (Timestamped)
            ├── "Why This Result?" Interactive Drawer
            └── Performance Telemetry (Inference ms, CPU, RAM)
```

---

## 2. Audio Streaming & Buffering Parameters

| Parameter | Recommended Default | Configurable Range | Rationale |
| :--- | :--- | :--- | :--- |
| **Incoming Chunk Size** | 85 ms (~1365 samples) | 20 ms – 100 ms | Determined by browser `ScriptProcessorNode` (4096 frames at 48kHz downsampled to 16kHz). |
| **Forensic Window Size**| 1000 ms (16,000 samples) | 250 ms – 3000 ms | Balances temporal context for phonetic resolution with low latency. |
| **Evaluation Step / Hop**| 400 ms (6,400 samples) | 200 ms – 1000 ms | Overlapping sliding analysis; ensures fresh predictions without saturating CPU. |
| **Max Ring Buffer** | 5000 ms (80,000 samples) | 3000 ms – 10000 ms| Bounded strictly in RAM; auto-trims older samples to guarantee privacy and zero memory leakage. |
| **Audio Format** | 16,000 Hz, 1-channel Mono, Float32 | Normalized [-1.0, 1.0] | Native Wav2Vec2 sample rate requirement; eliminates multi-channel phasing issues. |

---

## 3. Concurrency & Zero-Queue-Lag Strategy

### The Problem
When running deep neural models like Wav2Vec2 on standard CPU cores, inference takes **~150–250 ms**. If a WebSocket stream ingests audio chunks every **85 ms** and awaits inference synchronously, a massive backlog accumulates within seconds. After 30 seconds of speech, the model would be evaluating audio from 10 seconds in the past!

### The Solution: Non-Blocking Ring Buffer Slicing
1. **Unblocked Reception**: The async WebSocket receive loop (`await websocket.receive()`) only performs array concatenation to the in-memory ring buffer (`audio_buffer`). This operation takes `< 0.05 ms`.
2. **Backpressure Flag (`is_evaluating`)**:
   - When a cadence window expires, the server checks `if not is_evaluating`.
   - If `True`, `is_evaluating = True`, a copy of the **most recent** `current_window_samples` is sliced, and dispatched to Python's background thread pool via `asyncio.get_running_loop().run_in_executor`.
   - If `False` (previous inference still active), the cycle simply skips and continues buffering.
3. **Guaranteed Zero Lag**: The evaluator **always** evaluates the freshest audio window:
   `window_audio = audio_buffer[-current_window_samples:]`. Old audio is discarded without delaying the stream.

---

## 4. Voice Activity Detection (VAD) & Energy Gate

Before passing audio to feature extractors and neural layers, the system verifies whether meaningful human speech exists:
1. **Short-Time Energy / RMS**:
   $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}$$
2. **Frame-Level Voice Ratio**:
   The window is partitioned into 20 ms frames. A frame is considered voiced if $\text{RMS}_{\text{frame}} > 0.005$ and the Zero Crossing Rate (ZCR) falls within human speech boundaries ($0.02 < \text{ZCR} < 0.35$).
3. **Gate Decision**:
   - If `speech_ratio < 0.20`: The window is classified as ambient silence/noise. Inference is bypassed, returning `type: "silence_gate"` with state `ANALYZING / AWAITING SPEECH`.
   - If `speech_ratio >= 0.20`: Processing proceeds to the Quality Gate.

---

## 5. Audio Quality Gate (`app/quality.py`)

Degraded audio (packet loss, extreme clipping, high background noise) must never force a false REAL or FAKE classification. The Quality Gate evaluates:
- **Clipping Ratio**: Fraction of samples where $|x_i| \ge 0.99$. If $> 0.05$, penalty is applied.
- **Estimated SNR**: Peak-to-floor dynamic range estimate in dB.
- **Silence Ratio**: Percentage of sub-frames below acoustic noise threshold.
- **RMS Energy**: Minimum signal amplitude required for acoustic feature validity.

### Quality Score Formulation (0–100)
$$\text{Quality} = 100 - (\text{Clipping Penalty} + \text{Low Energy Penalty} + \text{Noise Floor Penalty})$$
- **Score $\ge 70$**: `EXCELLENT / GOOD` (Full forensic confidence).
- **Score 40–69**: `FAIR` (Evaluated with cautious confidence weighting).
- **Score $< 40$**: `POOR` $\rightarrow$ Triggers state `INCONCLUSIVE`.

---

## 6. Temporal Evidence Aggregator (`app/temporal.py`)

Real-world deepfake attacks persist across syllables; single-window anomalies can occur from microphone pops, coughs, or acoustic echoes. 

### Data Structure
Maintains a rolling FIFO queue of the last $K$ evaluated windows (default $K = 8$, representing ~3.2 seconds of historical context):
$$\mathcal{H} = [s_1, s_2, \dots, s_K]$$
where $s_i \in [0.0, 1.0]$ is the synthetic-class model score for window $i$.

### Aggregated Metrics
1. **Rolling Mean**: $\bar{s} = \frac{1}{K} \sum s_i$
2. **Rolling Median**: $\tilde{s} = \text{median}(\mathcal{H})$
3. **Elevated Window Ratio**:
   $$r_{\text{elevated}} = \frac{1}{K} \sum_{i=1}^K \mathbb{I}(s_i \ge 0.50)$$
4. **Persistent High-Risk Ratio**:
   $$r_{\text{critical}} = \frac{1}{K} \sum_{i=1}^K \mathbb{I}(s_i \ge 0.75)$$
5. **Score Variance**: $\sigma_s^2 = \frac{1}{K} \sum (s_i - \bar{s})^2$

---

## 7. Hysteresis State Machine (The 5 Live States)

To prevent the user interface from flickering erratically between risk levels, transitions into and out of high alert require consecutive confirmations:

```text
       ┌───────────────┐
       │ INITIALIZING  │
       └───────┬───────┘
               │ (Audio received, buffer primed)
               ▼
       ┌───────────────┐  Quality < 40   ┌──────────────┐
       │   ANALYZING   ├────────────────►│ INCONCLUSIVE │
       └───┬───────┬───┘                 └──────┬───────┘
           │       │                            │
   r_elev  │       │ r_elev >= 0.50             │ Quality restored
   < 0.35  │       │ (or 2 elevated windows)    │
           ▼       ▼                            ▼
      ┌────────┐ ┌────────────┐          ┌──────────────┐
      │LOW RISK│ │ SUSPICIOUS │◄─────────┤  ANALYZING   │
      └───┬────┘ └─────┬──────┘          └──────────────┘
          │            │
          │            │ 3 Consecutive Windows >= 0.75
          │            ▼
          │      ┌────────────┐
          │      │ HIGH RISK  │
          │      └─────┬──────┘
          │            │
          └────────────┘ (Requires 3 consecutive Low Windows to de-escalate)
```

### The 5 Primary States
1. **`INITIALIZING`**: Stream connected; buffering initial audio window.
2. **`ANALYZING`**: Active speech detected; evidence being accumulated.
3. **`LOW_RISK`**: Sustained natural vocal characteristics; authentic biological dynamics.
4. **`SUSPICIOUS`**: Elevated vocoder or neural synthetic signatures detected; monitoring closely.
5. **`HIGH_RISK`**: Persistent synthetic speech pattern confirmed over multiple consecutive windows.
*(Auxiliary State: **`INCONCLUSIVE`** when speech energy is too low, clipped, or distorted).*

---

## 8. Policy Arbitration Engine (`app/policy.py`)

The detector produces scientific evidence; the policy engine maps evidence into operational enterprise decisions:

| State / Condition | Action Code | Action Label | Recommended Enterprise Response |
| :--- | :--- | :--- | :--- |
| `LOW_RISK` | `ALLOW` | Allow Transaction / Call | Normal operation; zero friction. |
| `ANALYZING` | `MONITOR` | Active Passive Monitoring | Continue passive audio packet sampling. |
| `SUSPICIOUS` | `WARN_VERIFY` | Step-Up Verification | Display yellow warning; prompt agent or trigger out-of-band 2FA challenge. |
| `HIGH_RISK` | `ESCALATE` | Alert Security Operations | Trigger visual/auditory buzzer; flag session for immediate forensic audit. |
| Critical Persistent | `BLOCK_SIMULATION` | Simulated Call Termination | Log simulated disconnection event; **never drop live calls automatically**. |

---

## 9. Non-Negotiable Honesty & Transparent Disclaimers

1. **Synthetic-Class Model Score**: The neural output ($s \in [0, 1]$) represents the softmax probability of the synthetic class in the specific training distribution of `garystafford/wav2vec2-deepfake-voice-detector`. It is **NOT** a calibrated real-world probability of financial fraud or criminal intent.
2. **Supporting Acoustic Features**: LFCC, spectral centroid, spectral flatness, and phase statistics are acoustic vocoder anomaly metrics. They do **NOT** constitute a standalone trained classifier until a specific machine learning model is trained on them.
3. **Simulated Enforcement**: Real call drops or telephony disconnections are **simulated only** in this build.
