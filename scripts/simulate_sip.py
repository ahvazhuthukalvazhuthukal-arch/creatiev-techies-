"""Phase 17: Telephony WebRTC / SIP / Asterisk Policy Simulation.

NON-NEGOTIABLE HONESTY NOTICE:
This module demonstrates the policy arbitration and intervention state machine for
telecom integration. Real call termination MUST remain a controlled integration testbed
with Asterisk PBX / Kamailio, not an uncalibrated demo claim.
"""

import sys
import time
from typing import List, Dict, Any
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.scoring import calculate_risk

class TelecomPolicyAction:
    ALLOW = "ALLOW"
    MONITOR = "MONITOR"
    WARN_VERIFY = "WARN_VERIFY"
    ESCALATE = "ESCALATE"
    BLOCK_SIMULATION = "BLOCK_SIMULATION"

class CallInterventionStateMachine:
    def __init__(self, sustained_critical_threshold: int = 2):
        self.sustained_critical_threshold = sustained_critical_threshold
        self.consecutive_critical_count = 0
        self.call_history: List[Dict[str, Any]] = []

    def evaluate_window(self, window_id: int, synthetic_score: float) -> Dict[str, Any]:
        risk_res = calculate_risk(synthetic_score)
        risk = risk_res["risk_score"]
        
        if risk >= 75.0:
            self.consecutive_critical_count += 1
            if self.consecutive_critical_count >= self.sustained_critical_threshold:
                action = TelecomPolicyAction.BLOCK_SIMULATION
                intervention_reason = f"Critical deepfake score ({risk}%) sustained for {self.consecutive_critical_count} windows."
            else:
                action = TelecomPolicyAction.ESCALATE
                intervention_reason = f"High synthetic score ({risk}%) detected. Triggering supervisor verification."
        elif risk >= 50.0:
            self.consecutive_critical_count = 0
            action = TelecomPolicyAction.WARN_VERIFY
            intervention_reason = f"Suspicious score ({risk}%). Flagging call with forensic challenge prompt."
        elif risk >= 30.0:
            self.consecutive_critical_count = 0
            action = TelecomPolicyAction.MONITOR
            intervention_reason = f"Moderate / inconclusive score ({risk}%). Continuing passive acoustic telemetry."
        else:
            self.consecutive_critical_count = 0
            action = TelecomPolicyAction.ALLOW
            intervention_reason = f"Low risk ({risk}%). Normal call progression."

        event = {
            "window_id": window_id,
            "synthetic_score": round(synthetic_score, 4),
            "risk_score": risk,
            "verdict": risk_res["verdict"],
            "policy_action": action,
            "intervention_reason": intervention_reason,
            "telecom_carrier_signal": "SIP_BYE_DISPATCHED (SIMULATED)" if action == TelecomPolicyAction.BLOCK_SIMULATION else "AUDIO_PASSTHROUGH",
        }
        self.call_history.append(event)
        return event

def simulate_sample_call():
    print("=" * 75)
    print("VOCALGUARD AI V2 — SIP / ASTERISK TELEPHONY POLICY SIMULATION")
    print("=" * 75)
    print("Disclaimer: Simulated action stream. No live PBX is connected.")
    print("-" * 75)

    machine = CallInterventionStateMachine(sustained_critical_threshold=2)

    # Simulated sequence of 3-second streaming windows during a call:
    # 1: Normal greeting
    # 2: Normal conversation
    # 3: Synthesized voice clone injection begins
    # 4: Sustained clone attack
    simulated_stream = [
        (1, 0.08, "Caller speaks genuine greeting: 'Hello, I need to verify my account.'"),
        (2, 0.12, "Caller provides natural answers."),
        (3, 0.78, "Voice tone changes: Synthetic TTS clone injected for OTP transfer request."),
        (4, 0.94, "Synthetic clone continues: OTP extraction attempt."),
        (5, 0.88, "Synthetic clone persists."),
    ]

    for window_id, score, context in simulated_stream:
        event = machine.evaluate_window(window_id, score)
        print(f"Window #{event['window_id']} | Risk: {event['risk_score']:>5.1f}% | Verdict: {event['verdict']:<24}")
        print(f"  Context: {context}")
        print(f"  Action:  [{event['policy_action']}] -> {event['intervention_reason']}")
        print(f"  Signal:  {event['telecom_carrier_signal']}")
        print("-" * 75)

    print("\nPolicy Simulation Completed.")
    print("In a live Asterisk PBX, BLOCK_SIMULATION dispatches AMI/ARI 'ChannelHangup' command.")

if __name__ == "__main__":
    simulate_sample_call()
