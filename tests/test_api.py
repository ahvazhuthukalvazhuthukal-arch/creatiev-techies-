"""Integration and API tests for VocalGuard AI V2 FastAPI application."""

import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from app.main import app
from app.detector import VoiceDetector

client = TestClient(app)

def create_dummy_wav_bytes(duration: float = 1.0, freq: float = 440.0, sr: int = 16000) -> bytes:
    """Creates a short in-memory WAV byte stream."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["system"] == "VocalGuard AI V2"
    assert data["status"] == "online"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "device" in data
    assert "prototype_status" in data
    assert data["prototype_status"]["sip_asterisk"] is False
    assert data["prototype_status"]["onnx_int8"] is False

def test_live_demo_endpoint():
    response = client.get("/live-demo")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "VocalGuard AI V2" in response.text

def test_analyze_endpoint_valid_audio():
    wav_bytes = create_dummy_wav_bytes(duration=1.5, freq=500.0)
    files = {"file": ("test_tone.wav", wav_bytes, "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["filename"] == "test_tone.wav"
    assert data["sample_rate"] == 16000
    assert data["duration_seconds"] == 1.5

    # Model predictions
    assert "model" in data
    assert "synthetic_score" in data["model"]
    assert 0.0 <= data["model"]["synthetic_score"] <= 1.0

    # Forensic acoustic features
    assert "forensic_features" in data
    ff = data["forensic_features"]
    assert "spectral_flatness_mean" in ff
    assert "spectral_centroid_mean_hz" in ff
    assert "lfcc_mean" in ff
    assert "phase_diff_std" in ff

    # Risk assessment
    assert "risk" in data
    assert "risk_score" in data["risk"]
    assert data["risk"]["verdict"] in ["LOW RISK", "MODERATE / INCONCLUSIVE", "SUSPICIOUS", "CRITICAL RISK"]
    assert "not calibrated fraud probability" in data["risk"]["score_type"]

    # Timing metrics
    assert "timing" in data
    assert "model_inference_ms" in data["timing"]
    assert "total_analysis_ms" in data["timing"]

    # Prototype status flags
    assert data["prototype_status"]["sip_asterisk"] is False
    assert data["prototype_status"]["onnx_int8"] is False
    assert data["prototype_status"]["lightcnn_resnet"] is False
    assert data["prototype_status"]["rnnoise_wiener"] is False

def test_analyze_endpoint_empty_file():
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_analyze_endpoint_corrupt_file():
    files = {"file": ("corrupt.wav", b"not-a-valid-audio-file-header-corrupt", "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "failed to decode" in response.json()["detail"].lower()

def test_model_label_ambiguity_guard():
    """Verify that label parsing raises RuntimeError if synthetic class cannot be uniquely identified."""
    class FakeConfig:
        id2label = {0: "class_a", 1: "class_b"}

    class MockDetector:
        id2label = {0: "class_a", 1: "class_b"}
        _find_fake_index = VoiceDetector._find_fake_index

    mock = MockDetector()
    with pytest.raises(RuntimeError, match="Could not uniquely identify synthetic class"):
        mock._find_fake_index()
