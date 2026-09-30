"""Unit tests for app/preprocessing.py."""

import io
import numpy as np
import pytest
import soundfile as sf
from app.preprocessing import load_audio_bytes
from app.config import TARGET_SR

def create_in_memory_wav(samples: np.ndarray, sr: int) -> bytes:
    """Helper to create WAV bytes in-memory without disk persistence."""
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def test_load_audio_bytes_valid_mono():
    # 1 second of 16kHz sine wave
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_bytes = create_in_memory_wav(samples, 16000)

    audio, sr = load_audio_bytes(wav_bytes)
    assert sr == TARGET_SR
    assert audio.ndim == 1
    assert len(audio) == 16000
    assert audio.dtype == np.float32
    # Check normalized peak
    assert np.isclose(np.max(np.abs(audio)), 1.0, atol=1e-3)

def test_load_audio_bytes_stereo_to_mono():
    # 0.5s stereo at 16kHz
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    ch1 = 0.4 * np.sin(2 * np.pi * 440 * t)
    ch2 = 0.2 * np.sin(2 * np.pi * 880 * t)
    stereo = np.column_stack([ch1, ch2]).astype(np.float32)
    wav_bytes = create_in_memory_wav(stereo, 16000)

    audio, sr = load_audio_bytes(wav_bytes)
    assert sr == TARGET_SR
    assert audio.ndim == 1
    assert len(audio) == 8000

def test_load_audio_bytes_resampling():
    # 1 second at 44100 Hz resampled to 16000 Hz
    t = np.linspace(0, 1.0, 44100, endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_bytes = create_in_memory_wav(samples, 44100)

    audio, sr = load_audio_bytes(wav_bytes)
    assert sr == TARGET_SR
    # Length should be approximately 16000 samples
    assert abs(len(audio) - 16000) <= 10

def test_load_audio_bytes_silence():
    # 0.5s of pure silence
    samples = np.zeros(8000, dtype=np.float32)
    wav_bytes = create_in_memory_wav(samples, 16000)

    audio, sr = load_audio_bytes(wav_bytes)
    assert sr == TARGET_SR
    assert len(audio) == 8000
    assert np.all(audio == 0.0)

def test_load_audio_bytes_empty_rejection():
    with pytest.raises(ValueError, match="Empty audio file"):
        load_audio_bytes(b"")

def test_load_audio_bytes_invalid_corrupt():
    with pytest.raises(ValueError, match="Failed to decode audio file"):
        load_audio_bytes(b"this is corrupt random bytes not a wav")

def test_load_audio_bytes_max_duration_enforcement():
    # Attempting to load audio exceeding max duration
    t = np.linspace(0, 65.0, 65 * 8000, endpoint=False)
    samples = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_bytes = create_in_memory_wav(samples, 8000)

    with pytest.raises(ValueError, match="Audio exceeds maximum duration"):
        load_audio_bytes(wav_bytes, max_duration=60.0)
