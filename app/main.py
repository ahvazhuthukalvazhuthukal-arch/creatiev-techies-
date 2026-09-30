"""FastAPI application for VocalGuard AI V2.
Provides REST and WebSocket endpoints for audio deepfake inspection, forensic acoustic
feature extraction, transparent risk scoring, temporal evidence aggregation,
and real-time streaming forensic monitoring.
"""

from pathlib import Path
import time
import json
import asyncio
import numpy as np
import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import BASE_DIR, STATIC_DIR, AUDIO_SAMPLES_DIR, PROTOTYPE_STATUS, TARGET_SR, STREAMING_BUFFER_MAX_SECONDS
from .preprocessing import load_audio_bytes
from .detector import VoiceDetector
from .features import extract_all_features
from .scoring import calculate_risk
from .quality import compute_vad_and_energy, evaluate_audio_quality
from .temporal import TemporalAggregator, LiveForensicState
from .policy import arbitrate_policy, PolicyAction

app = FastAPI(
    title="VocalGuard AI V2",
    description="Real-Time AI-Powered Voice Cloning Detection & Forensic Prevention",
    version="2.0.0"
)

# Mount static folder if exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount audio samples folder for direct browser playback testing
if AUDIO_SAMPLES_DIR.exists():
    app.mount("/audio-samples", StaticFiles(directory=str(AUDIO_SAMPLES_DIR)), name="audio_samples")

# Instantiate detector singleton and warmup telemetry
detector = None
startup_metrics = {
    "cold_start_ms": 0.0,
    "warmup_ms": 0.0,
    "model_loaded": False,
    "warmed_up": False
}


def get_detector() -> VoiceDetector:
    global detector
    if detector is None:
        detector = VoiceDetector()
    return detector


@app.on_event("startup")
def startup_event():
    global startup_metrics
    t0 = time.perf_counter()
    det = get_detector()
    startup_metrics["cold_start_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    startup_metrics["model_loaded"] = True

    # Warmup forward pass
    warmup_ms = det.warmup(TARGET_SR, duration_seconds=1.0)
    startup_metrics["warmup_ms"] = warmup_ms
    startup_metrics["warmed_up"] = True


@app.get("/")
def root():
    return {
        "system": "VocalGuard AI V2",
        "status": "online",
        "version": "2.0.0",
        "docs_url": "/docs",
        "demo_url": "/live-demo"
    }


@app.get("/health")
def health():
    det = get_detector()
    return {
        "status": "ok",
        "model": det.model_name,
        "device": str(det.device),
        "synthetic_class": det.synthetic_label,
        "startup_telemetry": startup_metrics,
        "prototype_status": PROTOTYPE_STATUS
    }


@app.get("/live-demo")
def live_demo():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return FileResponse(index_file)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Primary file analysis endpoint.
    Processes audio completely in memory, runs Wav2Vec2 inference,
    extracts forensic features, evaluates signal quality, and computes transparent risk score.
    NEVER writes audio to persistent storage.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise ValueError("Uploaded file is empty")

        t_start = time.perf_counter()

        # 1. Preprocessing (Mono, 16kHz, Float32, Sanitized, Normalized)
        t_pre_start = time.perf_counter()
        audio, sr = load_audio_bytes(audio_bytes)
        preprocessing_ms = (time.perf_counter() - t_pre_start) * 1000.0

        # 2. Audio Quality Gate
        quality_res = evaluate_audio_quality(audio, sr)

        # 3. Model Forward Pass
        t_inf_start = time.perf_counter()
        det = get_detector()
        model_result = det.predict(audio, sr)
        inference_ms = (time.perf_counter() - t_inf_start) * 1000.0

        # 4. Forensic Acoustic Features
        t_feat_start = time.perf_counter()
        forensic_features = extract_all_features(audio, sr)
        feature_ms = (time.perf_counter() - t_feat_start) * 1000.0

        # 5. Transparent Risk Scoring (Multi-signal Ensemble)
        risk = calculate_risk(model_result["synthetic_score"], features=forensic_features)
        total_ms = (time.perf_counter() - t_start) * 1000.0

        # 6. Policy Determination
        policy = arbitrate_policy(
            state="LOW_RISK" if risk["risk_score"] < 40 else ("HIGH_RISK" if risk["risk_score"] >= 75 else "SUSPICIOUS"),
            synthetic_score=model_result["synthetic_score"],
            temporal_summary={"elevated_ratio": 1.0 if risk["risk_score"] >= 50 else 0.0, "consecutive_elevated": 1},
            quality_summary=quality_res,
        )

        return {
            "filename": file.filename,
            "sample_rate": sr,
            "duration_seconds": round(len(audio) / sr, 3),
            "quality": quality_res,
            "model": model_result,
            "forensic_features": forensic_features,
            "risk": risk,
            "policy": policy,
            "timing": {
                "preprocessing_ms": round(preprocessing_ms, 2),
                "model_inference_ms": round(inference_ms, 2),
                "feature_extraction_ms": round(feature_ms, 2),
                "total_analysis_ms": round(total_ms, 2),
            },
            "prototype_status": {
                "sip_asterisk": False,
                "onnx_int8": False,
                "lightcnn_resnet": False,
                "rnnoise_wiener": False
            },
            "disclaimer": "Prototype model score — not calibrated fraud probability. Acoustic indicators are supporting context."
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket, window_ms: int = Query(1000)):
    """High-performance real-time streaming endpoint with Zero-Queue-Lag architecture.
    Maintains a bounded rolling buffer of 16kHz PCM audio chunks.
    Dispatches inference non-blockingly to threadpool so WebSocket receive loop is never choked.
    Integrates VAD, Audio Quality Gate, Temporal Evidence Aggregation, and Policy Arbitration.
    """
    await websocket.accept()
    det = get_detector()

    # Bounded in-memory rolling buffer (max 5 seconds of 16kHz float32 audio)
    max_buffer_samples = int(STREAMING_BUFFER_MAX_SECONDS * TARGET_SR)
    audio_buffer = np.zeros(0, dtype=np.float32)

    current_window_samples = int((window_ms / 1000.0) * TARGET_SR)
    step_interval = 0.40  # Minimum 400ms cadence between evaluations
    is_evaluating = False
    last_eval_time = 0.0

    # Per-session Temporal Aggregator
    temporal = TemporalAggregator(
        max_history=8,
        elevated_threshold=0.50,
        high_risk_threshold=0.72,
        low_risk_threshold=0.38,
        consecutive_for_high=3,
        consecutive_for_low=3,
    )

    try:
        while True:
            message = await websocket.receive()

            # Support JSON configuration change or reset
            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    if "action" in payload and payload["action"] == "reset":
                        temporal.reset()
                        audio_buffer = np.zeros(0, dtype=np.float32)
                        await websocket.send_json({"type": "reset_ack", "status": "RESET_COMPLETE"})
                    if "window_ms" in payload:
                        new_ms = int(payload["window_ms"])
                        if 250 <= new_ms <= 5000:
                            current_window_samples = int((new_ms / 1000.0) * TARGET_SR)
                            await websocket.send_json({"type": "config_ack", "window_ms": new_ms})
                    if "step_ms" in payload:
                        new_step = float(payload["step_ms"]) / 1000.0
                        if 0.10 <= new_step <= 2.0:
                            step_interval = new_step
                except Exception:
                    pass

            # Audio chunk arrives as binary bytes (PCM 16-bit 16kHz)
            if "bytes" in message and message["bytes"]:
                chunk_bytes = message["bytes"]
                if len(chunk_bytes) == 0:
                    continue

                try:
                    samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                except Exception:
                    samples = np.frombuffer(chunk_bytes, dtype=np.float32)

                if len(samples) == 0:
                    continue

                # Fast append to rolling buffer (takes < 0.05 ms)
                audio_buffer = np.concatenate([audio_buffer, samples])
                if len(audio_buffer) > max_buffer_samples:
                    audio_buffer = audio_buffer[-max_buffer_samples:]

                now = time.perf_counter()
                # Trigger inference when buffer has enough audio, engine is not busy, and throttle interval met
                if len(audio_buffer) >= current_window_samples and not is_evaluating and (now - last_eval_time >= step_interval):
                    # ALWAYS take the most recent audio window (Zero-Queue-Lag)
                    window_audio = np.copy(audio_buffer[-current_window_samples:])

                    # Voice Activity & Energy Check (Step 5)
                    vad_info = compute_vad_and_energy(window_audio, TARGET_SR)

                    if not vad_info["speech_detected"]:
                        # Below speech energy / silence gate
                        await websocket.send_json({
                            "type": "silence_gate",
                            "status": "ANALYZING / AWAITING SPEECH",
                            "speech_detected": False,
                            "speech_ratio": vad_info["speech_ratio"],
                            "rms_energy": vad_info["rms_energy"],
                            "window_duration_seconds": round(len(window_audio) / TARGET_SR, 2)
                        })
                        continue

                    is_evaluating = True
                    last_eval_time = now

                    # Run inference in background executor thread to prevent event loop starvation
                    async def run_async_inference(audio_chunk, vad_meta):
                        nonlocal is_evaluating
                        try:
                            loop = asyncio.get_running_loop()

                            def compute():
                                t0 = time.perf_counter()

                                # Step 6: Audio Quality Gate
                                quality_res = evaluate_audio_quality(audio_chunk, TARGET_SR)

                                if not quality_res["is_sufficient"]:
                                    # Degraded audio -> Inconclusive
                                    temporal_summary = temporal.add_prediction(0.0, is_quality_sufficient=False)
                                    policy_decision = arbitrate_policy(
                                        LiveForensicState.INCONCLUSIVE,
                                        0.0,
                                        temporal_summary,
                                        quality_res
                                    )
                                    return {
                                        "quality": quality_res,
                                        "model": {"synthetic_score": 0.0, "status": "INCONCLUSIVE"},
                                        "forensic_features": {},
                                        "risk": {"risk_score": 0, "risk_level": "INCONCLUSIVE", "verdict": "Audio quality degraded / Inconclusive"},
                                        "temporal": temporal_summary,
                                        "policy": policy_decision,
                                        "t_inf": 0.0,
                                        "t_feat": 0.0,
                                        "t_total": (time.perf_counter() - t0) * 1000.0
                                    }

                                # Peak normalize for neural model
                                peak = float(np.max(np.abs(audio_chunk)))
                                norm = (audio_chunk / peak) if peak > 1e-8 else audio_chunk

                                # Step 7: Neural Model Inference
                                t_inf_start = time.perf_counter()
                                m_res = det.predict(norm, TARGET_SR)
                                t_inf = (time.perf_counter() - t_inf_start) * 1000.0

                                # Step 9: Supporting Forensic Acoustic Features
                                t_feat_start = time.perf_counter()
                                f_res = extract_all_features(norm, TARGET_SR)
                                t_feat = (time.perf_counter() - t_feat_start) * 1000.0

                                # Step 10: Multi-Signal Risk Scoring
                                r_res = calculate_risk(m_res["synthetic_score"], features=f_res)

                                # Step 11: Temporal Evidence Aggregation
                                t_summary = temporal.add_prediction(m_res["synthetic_score"], is_quality_sufficient=True)

                                # Step 18: Policy Arbitration
                                p_res = arbitrate_policy(
                                    t_summary["state"],
                                    m_res["synthetic_score"],
                                    t_summary,
                                    quality_res
                                )

                                t_total = (time.perf_counter() - t0) * 1000.0

                                return {
                                    "quality": quality_res,
                                    "model": m_res,
                                    "forensic_features": f_res,
                                    "risk": r_res,
                                    "temporal": t_summary,
                                    "policy": p_res,
                                    "t_inf": t_inf,
                                    "t_feat": t_feat,
                                    "t_total": t_total,
                                }

                            result = await loop.run_in_executor(None, compute)

                            # System Performance Telemetry (Step 20)
                            try:
                                cpu_pct = psutil.cpu_percent(interval=None)
                                mem_rss = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
                            except Exception:
                                cpu_pct = 0.0
                                mem_rss = 0.0

                            # Send structured versioned WebSocket response (Step 25)
                            await websocket.send_json({
                                "type": "analysis_result",
                                "timestamp": int(time.time() * 1000),
                                "window_ms": int((len(audio_chunk) / TARGET_SR) * 1000),
                                "step_ms": int(step_interval * 1000),
                                "status": result["temporal"]["state"],
                                "synthetic_score": round(result["model"].get("synthetic_score", 0.0), 4),
                                "risk_score": result["risk"].get("risk_score", 0),
                                "quality_score": result["quality"].get("quality_score", 0),
                                "quality": result["quality"],
                                "vad": vad_meta,
                                "model": result["model"],
                                "forensic_features": result["forensic_features"],
                                "risk": result["risk"],
                                "temporal": result["temporal"],
                                "policy": result["policy"],
                                "timing": {
                                    "model_inference_ms": round(result["t_inf"], 1),
                                    "feature_extraction_ms": round(result["t_feat"], 1),
                                    "total_window_analysis_ms": round(result["t_total"], 1),
                                },
                                "performance": {
                                    "cpu_percent": cpu_pct,
                                    "memory_rss_mb": mem_rss,
                                    "device": str(det.device),
                                },
                                "disclaimer": "Live stream model score — real-time sliding window. Supporting acoustic indicators are vocoder anomaly metrics, not a trained classifier."
                            })
                        except Exception as e:
                            try:
                                await websocket.send_json({"type": "error", "detail": str(e)})
                            except Exception:
                                pass
                        finally:
                            is_evaluating = False

                    asyncio.create_task(run_async_inference(window_audio, vad_info))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
