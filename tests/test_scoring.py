"""Unit tests for app/scoring.py (Risk Scoring & Thresholds)."""

import pytest
from app.scoring import calculate_risk

def test_risk_low_risk():
    res = calculate_risk(0.15)
    assert res["risk_score"] == 15.0
    assert res["verdict"] == "LOW RISK"
    assert "not calibrated fraud probability" in res["score_type"]

def test_risk_boundary_low_to_moderate():
    res_below = calculate_risk(0.299)
    assert res_below["verdict"] == "LOW RISK"

    res_exact = calculate_risk(0.30)
    assert res_exact["verdict"] == "MODERATE / INCONCLUSIVE"

def test_risk_moderate():
    res = calculate_risk(0.42)
    assert res["risk_score"] == 42.0
    assert res["verdict"] == "MODERATE / INCONCLUSIVE"

def test_risk_boundary_moderate_to_suspicious():
    res_below = calculate_risk(0.499)
    assert res_below["verdict"] == "MODERATE / INCONCLUSIVE"

    res_exact = calculate_risk(0.50)
    assert res_exact["verdict"] == "SUSPICIOUS"

def test_risk_suspicious():
    res = calculate_risk(0.68)
    assert res["risk_score"] == 68.0
    assert res["verdict"] == "SUSPICIOUS"

def test_risk_boundary_suspicious_to_critical():
    res_below = calculate_risk(0.749)
    assert res_below["verdict"] == "SUSPICIOUS"

    res_exact = calculate_risk(0.75)
    assert res_exact["verdict"] == "CRITICAL RISK"

def test_risk_critical():
    res = calculate_risk(0.95)
    assert res["risk_score"] == 95.0
    assert res["verdict"] == "CRITICAL RISK"

def test_risk_edge_cases():
    res_zero = calculate_risk(0.0)
    assert res_zero["risk_score"] == 0.0
    assert res_zero["verdict"] == "LOW RISK"

    res_one = calculate_risk(1.0)
    assert res_one["risk_score"] == 100.0
    assert res_one["verdict"] == "CRITICAL RISK"
