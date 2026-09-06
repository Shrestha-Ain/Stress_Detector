from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import json
import shutil
import os
import cv2
from starlette.concurrency import run_in_threadpool

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

def run_opencv_processing(temp_path: str):
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not read uploaded video file.")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Insert MediaPipe and math extraction here

    cap.release()
    return frame_count

@router.post("/process-video")
async def process_video_upload(video: UploadFile = File(...)):
    temp_path = f"temp_{video.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        frame_count = await run_in_threadpool(run_opencv_processing, temp_path)

        return {
            "status": "success",
            "frames_processed": frame_count,
            "metrics": {
                "hr_bpm": 72.5,
                "blink_rate": 16.2
            }
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.websocket("/ws/stream/{session_id}")
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
                "tracking_status": "Locked on Forehead ROI"
            })
    except WebSocketDisconnect:
        pass