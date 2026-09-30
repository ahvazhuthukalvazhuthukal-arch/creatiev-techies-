"""Pydantic schemas for VocalGuard AI V2 API requests and responses."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class RiskResult(BaseModel):
    risk_score: float = Field(..., description="Calculated UI risk score between 0.0 and 100.0")
    verdict: str = Field(..., description="Categorical risk band (LOW RISK, MODERATE / INCONCLUSIVE, SUSPICIOUS, CRITICAL RISK)")
    score_type: str = Field(
        default="synthetic-class model score; not calibrated fraud probability",
        description="Explicit description of what this score represents"
    )

class ModelResult(BaseModel):
    class_scores: Dict[str, float] = Field(..., description="Softmax probabilities across all classes")
    synthetic_score: float = Field(..., description="Raw probability assigned to the identified synthetic class")
    model_name: str
    device: str
    id2label: Dict[int, str]

class ForensicFeatures(BaseModel):
    spectral_flatness_mean: float
    spectral_flatness_std: float
    spectral_centroid_mean_hz: float
    spectral_centroid_std_hz: float
    lfcc_mean: float
    lfcc_std: float
    lfcc_abs_mean: float
    phase_mean: float
    phase_std: float
    phase_diff_std: float

class TimingBreakdown(BaseModel):
    model_inference_ms: float
    preprocessing_ms: Optional[float] = None
    feature_extraction_ms: Optional[float] = None
    total_analysis_ms: Optional[float] = None

class PrototypeStatus(BaseModel):
    sip_asterisk: bool = False
    onnx_int8: bool = False
    lightcnn_resnet: bool = False
    rnnoise_wiener: bool = False

class AnalyzeResponse(BaseModel):
    filename: str
    sample_rate: int
    duration_seconds: float
    model: ModelResult
    forensic_features: ForensicFeatures
    risk: RiskResult
    timing: TimingBreakdown
    prototype_status: PrototypeStatus
    disclaimer: str = "Prototype model score — not calibrated fraud probability. Acoustic indicators are supporting context."

class LiveStreamResult(BaseModel):
    window_duration_seconds: float
    model: ModelResult
    forensic_features: ForensicFeatures
    risk: RiskResult
    timing: TimingBreakdown
    disclaimer: str = "Live stream model score — experimental sliding window."
