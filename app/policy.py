"""Policy Arbitration Engine Module.
Decouples raw AI/acoustic evidence from enterprise enforcement actions.
Maps Live Forensic States to explicit actionable policies without automatically
terminating real-world telephony streams.
"""
from typing import Dict, Any
from app.temporal import LiveForensicState


class PolicyAction:
    ALLOW = "ALLOW"
    MONITOR = "MONITOR"
    WARN_VERIFY = "WARN_VERIFY"
    ESCALATE = "ESCALATE"
    BLOCK_SIMULATION = "BLOCK_SIMULATION"
    AUDIT_QUALITY = "AUDIT_QUALITY"


def arbitrate_policy(
    state: str,
    synthetic_score: float,
    temporal_summary: Dict[str, Any],
    quality_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluates multi-source evidence and assigns an enterprise action policy.
    
    IMPORTANT SAFETY RULE: Real phone calls or transactions are NEVER automatically
    blocked or terminated; actions are consultative, step-up verification, or simulated.
    """
    quality_sufficient = quality_summary.get("is_sufficient", True)
    consecutive_elevated = temporal_summary.get("consecutive_elevated", 0)
    elevated_ratio = temporal_summary.get("elevated_ratio", 0.0)

    if not quality_sufficient or state == LiveForensicState.INCONCLUSIVE:
        return {
            "action": PolicyAction.AUDIT_QUALITY,
            "action_label": "Request Better Audio / Quality Check",
            "recommended_response": "Audio quality degraded or low SNR; request customer to speak clearly or switch to cellular.",
            "is_blocking": False,
            "disclaimer": "Degraded audio prevents definitive biometric determination.",
        }

    if state == LiveForensicState.HIGH_RISK:
        if consecutive_elevated >= 4 and elevated_ratio >= 0.80:
            return {
                "action": PolicyAction.BLOCK_SIMULATION,
                "action_label": "Simulate Immediate Call Interception",
                "recommended_response": "Persistent synthetic anomalies detected across 4+ consecutive windows. Session flagged for high-priority fraud block.",
                "is_blocking": True,
                "simulated_only": True,
                "disclaimer": "Simulated intervention only. Real telephony disconnects require audited human authorization.",
            }
        return {
            "action": PolicyAction.ESCALATE,
            "action_label": "Escalate to Fraud Operations / Security Alert",
            "recommended_response": "Trigger silent supervisor chime, alert analyst terminal, and initiate secondary verification.",
            "is_blocking": False,
            "disclaimer": "Evidence indicates elevated vocoder/neural synthetic markers.",
        }

    if state == LiveForensicState.SUSPICIOUS:
        return {
            "action": PolicyAction.WARN_VERIFY,
            "action_label": "Step-Up Verification / Voice Challenge",
            "recommended_response": "Prompt caller with dynamic out-of-band knowledge challenge or SMS OTP confirmation.",
            "is_blocking": False,
            "disclaimer": "Moderate anomalies detected. Do not penalize user without step-up verification.",
        }

    if state == LiveForensicState.LOW_RISK:
        return {
            "action": PolicyAction.ALLOW,
            "action_label": "Allow Transaction / Normal Call Flow",
            "recommended_response": "Acoustic parameters and biological dynamics consistent with natural human vocalization.",
            "is_blocking": False,
            "disclaimer": "Low risk is not an absolute guarantee against zero-day voice clones.",
        }

    # Default for INITIALIZING / ANALYZING
    return {
        "action": PolicyAction.MONITOR,
        "action_label": "Passive Background Monitoring",
        "recommended_response": "Continue non-intrusive stream sampling and ring buffer accumulation.",
        "is_blocking": False,
        "disclaimer": "Accumulating sufficient speech frames for confident analysis.",
    }
