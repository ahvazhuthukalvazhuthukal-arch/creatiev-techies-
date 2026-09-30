import pytest
from app.policy import arbitrate_policy, PolicyAction
from app.temporal import LiveForensicState


def test_arbitrate_policy_allow():
    res = arbitrate_policy(
        state=LiveForensicState.LOW_RISK,
        synthetic_score=0.15,
        temporal_summary={"consecutive_elevated": 0, "elevated_ratio": 0.0},
        quality_summary={"is_sufficient": True}
    )
    assert res["action"] == PolicyAction.ALLOW
    assert res["is_blocking"] is False


def test_arbitrate_policy_monitor():
    res = arbitrate_policy(
        state=LiveForensicState.ANALYZING,
        synthetic_score=0.30,
        temporal_summary={"consecutive_elevated": 0, "elevated_ratio": 0.0},
        quality_summary={"is_sufficient": True}
    )
    assert res["action"] == PolicyAction.MONITOR
    assert res["is_blocking"] is False


def test_arbitrate_policy_warn_verify():
    res = arbitrate_policy(
        state=LiveForensicState.SUSPICIOUS,
        synthetic_score=0.65,
        temporal_summary={"consecutive_elevated": 2, "elevated_ratio": 0.50},
        quality_summary={"is_sufficient": True}
    )
    assert res["action"] == PolicyAction.WARN_VERIFY
    assert res["is_blocking"] is False


def test_arbitrate_policy_escalate():
    res = arbitrate_policy(
        state=LiveForensicState.HIGH_RISK,
        synthetic_score=0.85,
        temporal_summary={"consecutive_elevated": 3, "elevated_ratio": 0.60},
        quality_summary={"is_sufficient": True}
    )
    assert res["action"] == PolicyAction.ESCALATE
    assert res["is_blocking"] is False


def test_arbitrate_policy_block_simulation():
    res = arbitrate_policy(
        state=LiveForensicState.HIGH_RISK,
        synthetic_score=0.92,
        temporal_summary={"consecutive_elevated": 5, "elevated_ratio": 0.90},
        quality_summary={"is_sufficient": True}
    )
    assert res["action"] == PolicyAction.BLOCK_SIMULATION
    assert res["simulated_only"] is True


def test_arbitrate_policy_audit_quality():
    res = arbitrate_policy(
        state=LiveForensicState.INCONCLUSIVE,
        synthetic_score=0.0,
        temporal_summary={"consecutive_elevated": 0, "elevated_ratio": 0.0},
        quality_summary={"is_sufficient": False}
    )
    assert res["action"] == PolicyAction.AUDIT_QUALITY
    assert res["is_blocking"] is False
