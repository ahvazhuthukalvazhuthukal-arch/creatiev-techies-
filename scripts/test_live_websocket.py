"""Automated WebSocket Integration Test for VocalGuard AI V2 Live Endpoint.
Tests connection to ws://127.0.0.1:8080/ws/live, streaming audio frames,
and verifying silence_gate, analysis_result, config_ack, and reset_ack responses.
"""
import asyncio
import json
import sys
import numpy as np
import websockets

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


async def test_live_websocket():
    uri = "ws://127.0.0.1:8080/ws/live?window_ms=1000"
    print(f"Connecting to live WebSocket endpoint: {uri}...")

    async with websockets.connect(uri) as ws:
        print("[OK] Connected successfully to /ws/live!")

        # 1. Test Config Change
        print("\n1. Testing dynamic window reconfiguration...")
        await ws.send(json.dumps({"window_ms": 500, "step_ms": 200}))
        res = await asyncio.wait_for(ws.recv(), timeout=5.0)
        ack = json.loads(res)
        print(f"   Received: {ack}")
        assert ack.get("type") == "config_ack"
        assert ack.get("window_ms") == 500
        print("   [OK] Dynamic window configuration acknowledged!")

        # 2. Test Silence Gate
        print("\n2. Streaming silence frames (expecting silence_gate)...")
        silence_samples = np.zeros(2000, dtype=np.int16)
        # Send 4 chunks (~0.5s)
        for _ in range(5):
            await ws.send(silence_samples.tobytes())
            await asyncio.sleep(0.08)

        # Await response
        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        silence_res = json.loads(msg)
        print(f"   Received: {silence_res}")
        assert silence_res.get("type") == "silence_gate"
        assert silence_res.get("speech_detected") is False
        print("   [OK] Silence gate successfully intercepted low-energy frames!")

        # 3. Test Active Speech Evaluation
        print("\n3. Streaming simulated vocal harmonic frames (expecting analysis_result)...")
        sr = 16000
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        tone = ((0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 400 * t)) * 32767).astype(np.int16)
        
        # Send 15 chunks of speech (1.5 seconds)
        for _ in range(15):
            await ws.send(tone.tobytes())
            await asyncio.sleep(0.04)

        # Await analysis_result
        got_analysis = False
        for _ in range(10):
            try:
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                eval_res = json.loads(raw_msg)
                print(f"   Received event type: {eval_res.get('type')}")
                if eval_res.get("type") == "analysis_result":
                    print(f"   Status: {eval_res.get('status')}")
                    print(f"   Synthetic Score: {eval_res.get('synthetic_score')}")
                    print(f"   Quality Score: {eval_res.get('quality_score')} ({eval_res.get('quality', {}).get('quality_tier')})")
                    print(f"   Temporal State: {eval_res.get('temporal', {}).get('state')}")
                    print(f"   Policy Action: {eval_res.get('policy', {}).get('action')}")
                    print(f"   Timing: {eval_res.get('timing')}")
                    print(f"   Performance: {eval_res.get('performance')}")
                    assert "quality" in eval_res
                    assert "temporal" in eval_res
                    assert "policy" in eval_res
                    assert "timing" in eval_res
                    assert "performance" in eval_res
                    got_analysis = True
                    break
            except asyncio.TimeoutError:
                break

        assert got_analysis, "Did not receive analysis_result within timeout"
        print("   [OK] Full real-time analysis pipeline verified end-to-end!")

        # 4. Test Reset
        print("\n4. Testing state reset action...")
        await ws.send(json.dumps({"action": "reset"}))
        got_reset = False
        for _ in range(5):
            raw_reset = await asyncio.wait_for(ws.recv(), timeout=5.0)
            reset_msg = json.loads(raw_reset)
            if reset_msg.get("type") == "reset_ack":
                print(f"   Received: {reset_msg}")
                got_reset = True
                break
        assert got_reset, "Did not receive reset_ack"
        print("   [OK] Temporal state reset verified!")

    print("\n" + "=" * 60)
    print("ALL LIVE WEBSOCKET PIPELINE TESTS PASSED CLEANLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_live_websocket())
