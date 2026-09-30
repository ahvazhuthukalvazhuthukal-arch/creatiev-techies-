import numpy as np
import pytest
from app.quality import compute_vad_and_energy, evaluate_audio_quality


def test_compute_vad_and_energy_silence():
    silence = np.zeros(16000, dtype=np.float32)
    vad = compute_vad_and_energy(silence, 16000)
    assert vad["speech_detected"] is False
    assert vad["speech_ratio"] == 0.0
    assert vad["rms_energy"] == 0.0


def test_compute_vad_and_energy_speech():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 200 Hz tone + harmonics with realistic energy
    speech_sim = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)
    vad = compute_vad_and_energy(speech_sim.astype(np.float32), sr)
    assert vad["speech_detected"] is True
    assert vad["speech_ratio"] > 0.50
    assert vad["rms_energy"] > 0.10


def test_evaluate_audio_quality_clean():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    res = evaluate_audio_quality(signal, sr)
    assert res["quality_score"] >= 70
    assert res["quality_tier"] in ("GOOD", "EXCELLENT")
    assert res["is_sufficient"] is True
    assert res["metrics"]["clipping_ratio"] == 0.0


def test_evaluate_audio_quality_clipped():
    sr = 16000
    clipped = np.ones(8000, dtype=np.float32) * 1.0
    res = evaluate_audio_quality(clipped, sr)
    assert res["metrics"]["clipping_ratio"] == 1.0
    assert any("clipping" in r.lower() for r in res["reasons"])


def test_evaluate_audio_quality_silence():
    sr = 16000
    silence = np.zeros(8000, dtype=np.float32)
    res = evaluate_audio_quality(silence, sr)
    assert res["quality_score"] < 40
    assert res["quality_tier"] == "POOR"
    assert res["is_sufficient"] is False


def test_evaluate_audio_quality_empty():
    res = evaluate_audio_quality(np.zeros(0, dtype=np.float32), 16000)
    assert res["is_sufficient"] is False
    assert res["quality_score"] == 0
