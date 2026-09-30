"""Wav2Vec2 Deepfake Audio Detector module for VocalGuard AI V2.

Loads the pretrained acoustic classification model, verifies label assignments strictly,
and runs single-pass or streaming window neural inference.
"""

from typing import Dict, Any
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from .config import MODEL_NAME

class VoiceDetector:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load extractor and model
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        
        # Parse and verify labels strictly
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.label2id = {str(v): int(k) for k, v in self.model.config.id2label.items()}
        self.fake_index = self._find_fake_index()
        self.synthetic_label = self.id2label[self.fake_index]

    def _find_fake_index(self) -> int:
        """Find the synthetic class ONLY from actual labels using canonical candidates:
        fake, synthetic, spoof, deepfake, generated.
        
        If there is not exactly one match, fail loudly and require explicit configuration.
        Do not guess.
        """
        candidates = {"fake", "synthetic", "spoof", "deepfake", "generated"}
        matches = []
        for idx, label in self.id2label.items():
            normalized = label.lower().replace("_", " ").replace("-", " ")
            # Check individual word tokens or substring match in normalized label
            tokens = set(normalized.split())
            if tokens.intersection(candidates) or any(x in normalized for x in candidates):
                matches.append(idx)
                
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely identify synthetic class from labels {self.id2label}. "
                f"Found {len(matches)} matching candidates: {matches}. Explicit configuration required."
            )
        return matches[0]

    def predict(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Runs neural forward pass on audio samples.
        
        Args:
            audio: 1D float32 numpy array of resampled audio.
            sr: Sampling rate (expected 16000 Hz).
            
        Returns:
            Dictionary with class_scores, synthetic_score, model_name, device, and id2label.
        """
        if len(audio) == 0:
            raise ValueError("Cannot predict on empty audio array")
            
        inputs = self.extractor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        scores = probs.detach().cpu().numpy()
        
        return {
            "class_scores": {self.id2label.get(i, f"class_{i}"): float(v) for i, v in enumerate(scores)},
            "synthetic_score": float(scores[self.fake_index]),
            "model_name": self.model_name,
            "device": str(self.device),
            "id2label": self.id2label
        }

    def warmup(self, sr: int = 16000, duration_seconds: float = 1.0) -> float:
        """Executes a warm-up forward pass on dummy audio to prime PyTorch kernels and JIT cache.
        Returns warmup latency in milliseconds.
        """
        import time
        dummy = np.zeros(int(sr * duration_seconds), dtype=np.float32)
        t0 = time.perf_counter()
        _ = self.predict(dummy, sr)
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        return float(round(warmup_ms, 2))
