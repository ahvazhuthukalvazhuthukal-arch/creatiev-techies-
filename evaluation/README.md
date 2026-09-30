# VocalGuard AI V2 — Evaluation Framework & Dataset Protocol

## Non-Negotiable Scientific Honesty

1. **Smoke-Test Sets vs Production Benchmarks**:
   - The files provided in `audio_samples/` are purely software integration fixtures to verify that audio pipelines, feature extractors, and metrics calculations run without exceptions.
   - **Never cite scores from local smoke fixtures as real-world operational accuracy.**

2. **Representative Evaluation Protocol**:
   To establish legitimate deepfake voice detection benchmarks, evaluation must be conducted on recognized standard corpuses:
   - **ASVspoof 2019 / 2021 (LA & DF tracks)**: Logical Access and Deepfake evaluations including state-of-the-art TTS, VC (voice conversion), and vocoder algorithms.
   - **In-The-Wild Audio Deepfake Dataset**: Real-world recorded speech clips collected from YouTube, podcasts, and social media.
   - **WaveFake Dataset**: Multi-architecture synthetic speech generated across 6+ neural vocoders (MelGAN, HiFi-GAN, WaveGlow, Parallel WaveGAN, etc.).

3. **Evaluation Protocol Requirements**:
   - **Speaker-Disjoint Splits**: Test speakers must never appear in the training or fine-tuning set.
   - **Codec & Channel Variation**: Audio must be tested across telephony codecs (G.711 A-law/μ-law, AMR-WB, Opus) and bitrates (8 kbps to 64 kbps).
   - **Unseen Synthesis Systems**: Models must be evaluated against vocoders and TTS architectures not seen during training.
   - **Environmental Noise & Reverberation**: Performance must be evaluated with SNR ranging from 0 dB to 20 dB (street noise, office chatter, packet jitter).

## Running Evaluation

```bash
python scripts/generate_sample_audio.py
python evaluation/evaluate.py
```

### Metrics Reported:
- **EER (Equal Error Rate)**: Point where False Acceptance Rate equals False Rejection Rate.
- **min t-DCF**: Minimum tandem Detection Cost Function (when paired with ASV).
- **Confusion Matrix**: True Positives, True Negatives, False Positives, False Negatives.
- **Precision, Recall, F1-Score**.
