"""Configuration settings for VocalGuard AI V2.
Enforces non-negotiable honesty regarding model parameters, target architecture, and evaluation boundaries.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
AUDIO_SAMPLES_DIR = BASE_DIR / "audio_samples"

# Audio Constraints
TARGET_SR = 16000
MAX_DURATION_SECONDS = 60
STREAMING_BUFFER_MAX_SECONDS = 5.0

# Supported Sliding Window Durations for Latency & Sensitivity Experiments (in ms)
SUPPORTED_WINDOW_SIZES_MS = [250, 500, 750, 1000, 3000]
DEFAULT_STREAM_WINDOW_MS = 3000

# Model Configuration
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"

# Transparent UI Risk Score Thresholds (Heuristic Bands - Not Calibrated Fraud Probabilities)
THRESHOLD_CRITICAL = 75.0
THRESHOLD_SUSPICIOUS = 50.0
THRESHOLD_MODERATE = 30.0

# Prototype Capabilities Checklist
PROTOTYPE_STATUS = {
    "wav2vec2_classifier": True,
    "forensic_features_spectral": True,
    "forensic_features_lfcc_experimental": True,
    "forensic_features_phase_experimental": True,
    "websocket_streaming_pipeline": True,
    "sip_asterisk": False,
    "onnx_int8": False,
    "lightcnn_resnet": False,
    "rnnoise_wiener": False,
}
