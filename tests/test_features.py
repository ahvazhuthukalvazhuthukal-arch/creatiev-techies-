"""Unit tests for app/features.py (Acoustic and Forensic features)."""

import numpy as np
import pytest
from app.features import spectral_features, lfcc_features, phase_features, extract_all_features

def test_spectral_features_sine():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 1000 Hz pure sine wave
    sine = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    feats = spectral_features(sine, sr)

    assert "spectral_flatness_mean" in feats
    assert "spectral_centroid_mean_hz" in feats
    # Pure sine wave has low spectral flatness (~0)
    assert feats["spectral_flatness_mean"] < 0.05
    # Centroid should be in the neighborhood of 1000 Hz
    assert 800 <= feats["spectral_centroid_mean_hz"] <= 1200

def test_spectral_features_noise():
    sr = 16000
    np.random.seed(42)
    noise = np.random.uniform(-0.5, 0.5, sr).astype(np.float32)
    feats = spectral_features(noise, sr)

    # White noise has much higher spectral flatness than a tone
    assert feats["spectral_flatness_mean"] > 0.1

def test_lfcc_features():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
    feats = lfcc_features(sine, sr, n_lfcc=20)

    assert "lfcc_mean" in feats
    assert "lfcc_std" in feats
    assert "lfcc_abs_mean" in feats
    assert isinstance(feats["lfcc_mean"], float)
    assert not np.isnan(feats["lfcc_mean"])

def test_phase_features():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    feats = phase_features(sine, sr)

    assert "phase_mean" in feats
    assert "phase_std" in feats
    assert "phase_diff_std" in feats
    assert not np.isnan(feats["phase_diff_std"])

def test_extract_all_features():
    sr = 16000
    audio = np.zeros(16000, dtype=np.float32)
    all_feats = extract_all_features(audio, sr)

    expected_keys = {
        "spectral_flatness_mean", "spectral_flatness_std",
        "spectral_centroid_mean_hz", "spectral_centroid_std_hz",
        "lfcc_mean", "lfcc_std", "lfcc_abs_mean",
        "phase_mean", "phase_std", "phase_diff_std"
    }
    assert expected_keys.issubset(set(all_feats.keys()))
