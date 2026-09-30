import pytest
from app.temporal import TemporalAggregator, LiveForensicState


def test_temporal_aggregator_init():
    agg = TemporalAggregator(max_history=5)
    summary = agg.get_summary()
    assert summary["state"] == LiveForensicState.INITIALIZING
    assert summary["history_length"] == 0
    assert summary["rolling_mean"] == 0.0


def test_temporal_aggregator_low_risk_flow():
    agg = TemporalAggregator(max_history=5)
    # Feed 3 natural windows
    agg.add_prediction(0.12, is_quality_sufficient=True)
    summary = agg.add_prediction(0.18, is_quality_sufficient=True)
    assert summary["state"] == LiveForensicState.LOW_RISK
    assert summary["rolling_mean"] < 0.25
    assert summary["elevated_windows"] == 0


def test_temporal_aggregator_suspicious_flow():
    agg = TemporalAggregator(max_history=5)
    agg.add_prediction(0.20, is_quality_sufficient=True)
    agg.add_prediction(0.55, is_quality_sufficient=True)
    summary = agg.add_prediction(0.60, is_quality_sufficient=True)
    assert summary["state"] == LiveForensicState.SUSPICIOUS
    assert summary["elevated_windows"] >= 2


def test_temporal_aggregator_high_risk_hysteresis():
    agg = TemporalAggregator(max_history=5, consecutive_for_high=3, consecutive_for_low=3)
    # First 2 high windows -> SUSPICIOUS
    agg.add_prediction(0.85, is_quality_sufficient=True)
    summary = agg.add_prediction(0.90, is_quality_sufficient=True)
    assert summary["state"] == LiveForensicState.SUSPICIOUS
    assert summary["consecutive_elevated"] == 2

    # 3rd consecutive high window -> HIGH_RISK
    summary = agg.add_prediction(0.92, is_quality_sufficient=True)
    assert summary["state"] == LiveForensicState.HIGH_RISK

    # 1 low window does NOT immediately drop HIGH_RISK (Hysteresis)
    summary = agg.add_prediction(0.15, is_quality_sufficient=True)
    assert summary["state"] == LiveForensicState.HIGH_RISK

    # 2nd low window still dampens
    agg.add_prediction(0.12, is_quality_sufficient=True)
    
    # 3rd consecutive low window exits HIGH_RISK
    summary = agg.add_prediction(0.10, is_quality_sufficient=True)
    assert summary["state"] != LiveForensicState.HIGH_RISK


def test_temporal_aggregator_inconclusive():
    agg = TemporalAggregator(max_history=5)
    summary = agg.add_prediction(0.50, is_quality_sufficient=False)
    assert summary["state"] == LiveForensicState.INCONCLUSIVE


def test_temporal_aggregator_reset():
    agg = TemporalAggregator(max_history=5)
    agg.add_prediction(0.80, is_quality_sufficient=True)
    agg.reset()
    assert agg.current_state == LiveForensicState.INITIALIZING
    assert len(agg.history) == 0
