"""Audio preprocessing module for VocalGuard AI V2.
Handles decoding, mono conversion, 16kHz resampling, NaN/Inf sanitization,
peak normalization, and constraint checks without saving any persistent audio to disk.
"""

import io
import numpy as np
import librosa
import soundfile as sf
from .config import TARGET_SR, MAX_DURATION_SECONDS

def load_audio_bytes(audio_bytes: bytes, target_sr: int = TARGET_SR, max_duration: float = MAX_DURATION_SECONDS):
    """Decodes raw audio bytes entirely in-memory.

    Args:
        audio_bytes: In-memory byte buffer of an audio file (e.g. WAV, FLAC, OGG).
        target_sr: Desired sampling rate (default: 16000 Hz).
        max_duration: Maximum allowed audio length in seconds (default: 60s).

    Returns:
        tuple (audio_samples, sample_rate) where audio_samples is a 1D float32 numpy array.

    Raises:
        ValueError: If audio is empty, invalid, exceeds max duration, or contains no usable samples.
    """
    if not audio_bytes:
        raise ValueError("Empty audio file")

    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as exc:
        raise ValueError(f"Failed to decode audio file: {exc}") from exc

    # Convert multi-channel to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Cast to float32
    audio = np.asarray(audio, dtype=np.float32)

    # Sanitize any NaN or Infinite values
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Check for empty sample array
    if len(audio) == 0:
        raise ValueError("No samples found")

    # Duration boundary check
    duration = len(audio) / target_sr
    if duration > max_duration:
        raise ValueError(f"Audio exceeds maximum duration of {max_duration} seconds (received {duration:.1f}s)")

    # Safe peak normalization
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-8:
        audio = audio / peak

    return audio.astype(np.float32), target_sr
