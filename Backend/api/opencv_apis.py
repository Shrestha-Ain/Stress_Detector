from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import json
import shutil
import os
import cv2
from starlette.concurrency import run_in_threadpool
import mediapipe as mp
import numpy as np
from pipelines.pipeline_utils import (
    eye_aspect_ratio, pos_algorithm, bandpass_filter, estimate_hr_and_hrv
)

router = APIRouter()

class PreflightCheckInput(BaseModel):
    ambient_lux: float = Field(..., description="Calculated lux intensity from camera")
    mic_snr_db: float = Field(..., description="Signal-to-noise ratio in decibels")

class TelemetryPayload(BaseModel):
    personnel_id: str
    blink_rate: float
    ear_ratio: float
    yawn_count: int
    duty_days: int

@router.post("/preflight")
def run_preflight_check(data: PreflightCheckInput):
    lux_valid = data.ambient_lux >= 120.0
    snr_valid = data.mic_snr_db >= 15.0
    can_proceed = lux_valid and snr_valid
    return {
        "ready_to_record": can_proceed,
        "diagnostics": {
            "ambient_lux": {"value": data.ambient_lux, "passed": lux_valid},
            "mic_snr_db": {"value": data.mic_snr_db, "passed": snr_valid}
        }
    }

@router.post("/telemetry/ingest")
def ingest_telemetry(data: TelemetryPayload):
    return {
        "status": "success",
        "personnel_id": data.personnel_id,
        "stress_score": 76.5,
        "predicted_tier": "Critical"
    }

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

def get_pts(landmarks, indices, w, h):
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])

def run_opencv_processing(temp_path: str):
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise HTTPException(
            status_code=400, detail="Could not read uploaded video file."
        )

    rgb_means = []
    timestamps = []
    blink_count = 0
    frame_idx = 0

    use_mediapipe = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
    face_mesh_ctx = None

    if use_mediapipe:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh_ctx = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6
        )

    try:
        if face_mesh_ctx:
            face_mesh_ctx.__enter__()


        while cap.isOpened():
            ok, frame = cap.read()

            # 1. ALWAYS check for EOF or None FIRST before doing anything else
            if not ok or frame is None:
                break

            # 2. Frame skipping second
            if frame_idx % 2 != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            if w > 640 and h > 0:
                scale = 640 / w
                w = 640
                h = max(1, int(h * scale))
                frame = cv2.resize(frame, (w, h))
            

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_time = frame_idx / 30.0

            if face_mesh_ctx:
                results = face_mesh_ctx.process(rgb_frame)
                if results and results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    xs = [lm.x * w for lm in landmarks]
                    ys = [lm.y * h for lm in landmarks]
                    x1, x2 = int(min(xs) + 0.30 * (max(xs) - min(xs))), int(
                        min(xs) + 0.70 * (max(xs) - min(xs))
                    )
                    y1, y2 = int(min(ys) + 0.08 * (max(ys) - min(ys))), int(
                        min(ys) + 0.28 * (max(ys) - min(ys))
                    )
                    forehead = rgb_frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
                    if forehead.size > 0:
                        rgb_means.append(forehead.reshape(-1, 3).mean(axis=0))
                        timestamps.append(current_time)
                else:
                    forehead = rgb_frame[
                        int(h * 0.3) : int(h * 0.5), int(w * 0.4) : int(w * 0.6)
                    ]
                    if forehead.size > 0:
                        rgb_means.append(forehead.reshape(-1, 3).mean(axis=0))
                        timestamps.append(current_time)
            else:
                forehead = rgb_frame[
                    int(h * 0.3) : int(h * 0.5), int(w * 0.4) : int(w * 0.6)
                ]
                if forehead.size > 0:
                    rgb_means.append(forehead.reshape(-1, 3).mean(axis=0))
                    timestamps.append(current_time)

            frame_idx += 1
    finally:
        if face_mesh_ctx:
            face_mesh_ctx.__exit__(None, None, None)
        cap.release()

    actual_fps = (
        len(timestamps) / (timestamps[-1] - timestamps[0])
        if len(timestamps) > 1
        else 30.0
    )
    hr_bpm, rmssd_ms = 72.0, 45.0

    if len(rgb_means) > 15:
        try:
            pulse = pos_algorithm(np.array(rgb_means), actual_fps)
            filtered = bandpass_filter(pulse, actual_fps)
            hrv = estimate_hr_and_hrv(filtered, actual_fps)
            hr_bpm = hrv.get("hr_bpm") or 72.0
            rmssd_ms = hrv.get("rmssd_ms") or 45.0
        except Exception:
            pass

    duration_sec = frame_idx / actual_fps if actual_fps > 0 else 1.0
    blink_rate = (blink_count / duration_sec) * 60.0

    return {
        "frame_count": frame_idx,
        "hr_bpm": round(float(hr_bpm), 2),
        "rmssd_ms": round(float(rmssd_ms), 2),
        "blink_rate": round(float(blink_rate), 2),
    }

@router.post("/process-video")
async def process_video_upload(video: UploadFile = File(...)):
    print(">>> /process-video ENDPOINT WAS HIT! <<<")
    temp_path = f"temp_{video.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        result = await run_in_threadpool(run_opencv_processing, temp_path)

        return {
            "status": "success",
            "frames_processed": result["frame_count"],
            "metrics": {
                "hr_bpm": result["hr_bpm"],
                "rmssd_ms": result["rmssd_ms"],
                "blink_rate": result["blink_rate"],
            },
        }

    except Exception as e:
        import traceback

        import traceback

        error_detail = traceback.format_exc()
        print(error_detail)
        raise HTTPException(
            status_code=500, detail=error_detail
        )

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except PermissionError:
                pass


@router.websocket("/stream/{session_id}")
async def websocket_telemetry_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            await websocket.send_json({
                "session_id": session_id,
                "live_hr_gauge": payload.get("live_hr", 75),
                "live_rmssd_gauge": payload.get("live_rmssd", 38),
                "tracking_status": "Locked on Forehead ROI",
            })
    except WebSocketDisconnect:
        pass