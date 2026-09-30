"""Temporal Evidence Aggregation & Hysteresis State Machine Module.
Aggregates rolling inference windows to eliminate single-packet jitter and false positives.
Implements the 5 Live Forensic States with hysteresis damping.
"""
from typing import List, Dict, Any, Optional
import numpy as np


class LiveForensicState:
    INITIALIZING = "INITIALIZING"
    ANALYZING = "ANALYZING"
    LOW_RISK = "LOW_RISK"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"
    INCONCLUSIVE = "INCONCLUSIVE"


class TemporalAggregator:
    """Maintains a bounded historical sliding queue of model scores and acoustic metrics
    to compute smoothed temporal evidence and manage stable state transitions.
    """

    def __init__(
        self,
        max_history: int = 8,
        elevated_threshold: float = 0.50,
        high_risk_threshold: float = 0.72,
        low_risk_threshold: float = 0.38,
        consecutive_for_high: int = 3,
        consecutive_for_low: int = 3,
    ):
        self.max_history = max(max_history, 3)
        self.elevated_threshold = elevated_threshold
        self.high_risk_threshold = high_risk_threshold
        self.low_risk_threshold = low_risk_threshold
        self.consecutive_for_high = consecutive_for_high
        self.consecutive_for_low = consecutive_for_low

        self.history: List[Dict[str, Any]] = []
        self.current_state = LiveForensicState.INITIALIZING
        self._consecutive_elevated = 0
        self._consecutive_low = 0
        self.total_evaluations = 0

    def add_prediction(
        self,
        synthetic_score: float,
        is_quality_sufficient: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Appends a new window evaluation to history and updates the state machine."""
        self.total_evaluations += 1
        record = {
            "index": self.total_evaluations,
            "synthetic_score": float(np.clip(synthetic_score, 0.0, 1.0)),
            "is_quality_sufficient": bool(is_quality_sufficient),
            "metadata": metadata or {},
        }
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Handle Inconclusive quality
        if not is_quality_sufficient:
            self.current_state = LiveForensicState.INCONCLUSIVE
            return self.get_summary()

        score = record["synthetic_score"]

        # Track consecutive runs for hysteresis
        if score >= self.high_risk_threshold or score >= self.elevated_threshold:
            self._consecutive_elevated += 1
            self._consecutive_low = 0
        elif score <= self.low_risk_threshold:
            self._consecutive_low += 1
            self._consecutive_elevated = 0
        else:
            # Neutral zone resets neither aggressively
            pass

        # State transition logic
        scores = [r["synthetic_score"] for r in self.history if r["is_quality_sufficient"]]
        if len(scores) < 2:
            self.current_state = LiveForensicState.ANALYZING
        else:
            mean_score = float(np.mean(scores))
            elevated_count = sum(1 for s in scores if s >= self.elevated_threshold)
            elevated_ratio = elevated_count / len(scores)

            if self.current_state == LiveForensicState.HIGH_RISK:
                # To exit HIGH_RISK, require persistent low windows
                if self._consecutive_low >= self.consecutive_for_low or (mean_score < self.low_risk_threshold and elevated_ratio <= 0.20):
                    self.current_state = LiveForensicState.LOW_RISK if mean_score < self.low_risk_threshold else LiveForensicState.SUSPICIOUS
            else:
                # Entering HIGH_RISK requires consecutive elevated windows OR strong majority across at least consecutive_for_high windows
                if (self._consecutive_elevated >= self.consecutive_for_high) or (len(scores) >= self.consecutive_for_high and elevated_ratio >= 0.70 and mean_score >= self.high_risk_threshold):
                    self.current_state = LiveForensicState.HIGH_RISK
                elif (elevated_ratio >= 0.35) or (score >= self.elevated_threshold) or (mean_score >= 0.48):
                    self.current_state = LiveForensicState.SUSPICIOUS
                else:
                    self.current_state = LiveForensicState.LOW_RISK

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Calculates current rolling statistics and temporal evidence metrics."""
        scores = [r["synthetic_score"] for r in self.history if r["is_quality_sufficient"]]
        
        if not scores:
            return {
                "state": self.current_state,
                "history_length": 0,
                "rolling_mean": 0.0,
                "rolling_median": 0.0,
                "rolling_max": 0.0,
                "score_variance": 0.0,
                "elevated_windows": 0,
                "elevated_ratio": 0.0,
                "high_risk_windows": 0,
                "high_risk_ratio": 0.0,
                "consecutive_elevated": self._consecutive_elevated,
                "consecutive_low": self._consecutive_low,
                "temporal_evidence_level": "INSUFFICIENT_DATA",
            }

        mean_val = float(np.mean(scores))
        median_val = float(np.median(scores))
        max_val = float(np.max(scores))
        var_val = float(np.var(scores)) if len(scores) > 1 else 0.0
        elevated_count = sum(1 for s in scores if s >= self.elevated_threshold)
        elevated_ratio = float(elevated_count / len(scores))
        high_risk_count = sum(1 for s in scores if s >= self.high_risk_threshold)
        high_risk_ratio = float(high_risk_count / len(scores))

        # Temporal evidence level categorization
        if self.current_state == LiveForensicState.HIGH_RISK:
            evidence_level = "CRITICAL_PERSISTENT"
        elif elevated_ratio >= 0.50 or mean_val >= self.elevated_threshold:
            evidence_level = "ELEVATED"
        elif elevated_ratio > 0.0:
            evidence_level = "MODERATE"
        else:
            evidence_level = "NATURAL_CONSISTENT"

        return {
            "state": self.current_state,
            "history_length": len(scores),
            "rolling_mean": round(mean_val, 4),
            "rolling_median": round(median_val, 4),
            "rolling_max": round(max_val, 4),
            "score_variance": round(var_val, 5),
            "elevated_windows": elevated_count,
            "elevated_ratio": round(elevated_ratio, 3),
            "high_risk_windows": high_risk_count,
            "high_risk_ratio": round(high_risk_ratio, 3),
            "consecutive_elevated": self._consecutive_elevated,
            "consecutive_low": self._consecutive_low,
            "temporal_evidence_level": evidence_level,
        }

    def reset(self):
        """Clears temporal queue and resets to INITIALIZING state."""
        self.history.clear()
        self.current_state = LiveForensicState.INITIALIZING
        self._consecutive_elevated = 0
        self._consecutive_low = 0
        self.total_evaluations = 0
