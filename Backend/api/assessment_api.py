from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from datetime import datetime, timezone
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
import shutil
import os

from database import get_db
from models_db import gen_id
from api.auth_api import require_role, get_current_user
from pipelines.video_processing import run_opencv_processing, run_audio_processing_from_video
from pipelines.pipeline_utils import score_stress

router = APIRouter()


class InterventionPayload(BaseModel):
    personnel_id: str
    action_type: str
    notes: str


# ---------------------------------------------------------
# The candidate app's one real endpoint
# ---------------------------------------------------------
@router.post("/full-evaluate")
async def full_evaluate(
    video: UploadFile = File(...),
    duty_hours_streak: float = Form(...),
    relax_hours_preceding: float = Form(...),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    JWT (Authorization header, from /login) + one video file + the two
    numbers the candidate types in on the same screen right before
    recording (hours on duty, hours rested). personnel_id comes from the
    token — nothing else needs to be sent separately.
    """
    personnel_id = current_user["personnel_id"]

    video_temp = f"temp_{video.filename}"
    with open(video_temp, "wb") as f:
        shutil.copyfileobj(video.file, f)
    try:
        # Both run against the same uploaded file: run_opencv_processing reads
        # the video frames, run_audio_processing_from_video demuxes and reads
        # the audio track embedded in it — no second upload needed.
        video_metrics = await run_in_threadpool(run_opencv_processing, video_temp)
        voice_metrics = await run_in_threadpool(run_audio_processing_from_video, video_temp)
    finally:
        if os.path.exists(video_temp):
            os.remove(video_temp)

    features = {
        "hr_bpm": video_metrics["hr_bpm"],
        "rmssd_ms": video_metrics["rmssd_ms"],
        "blink_rate_bpm": video_metrics["blink_rate"],
        "brow_ratio": video_metrics["brow_ratio"],
        "pitch_mean_hz": voice_metrics["pitch_mean_hz"],
        "pitch_std_hz": voice_metrics["pitch_std_hz"],
        "duty_hours_streak": duty_hours_streak,
        "relax_hours_preceding": relax_hours_preceding,
    }

    result = score_stress(features)

    session_doc = {
        "_id": gen_id(),
        "personnel_id": personnel_id,
        "duty_hours_streak": duty_hours_streak,
        "relax_hours_preceding": relax_hours_preceding,
        "hr_bpm": features["hr_bpm"],
        "rmssd_ms": features["rmssd_ms"],
        "blink_rate_bpm": features["blink_rate_bpm"],
        "brow_ratio": features["brow_ratio"],
        "head_jitter": video_metrics["head_jitter"],  # logged for dashboard/history, not scored yet
        "pitch_mean_hz": features["pitch_mean_hz"],
        "pitch_std_hz": features["pitch_std_hz"],
        "stress_probability": result["stress_probability"],
        "classification": result["classification"],
        "shap_attribution": result["shap_attribution"],
        "created_at": datetime.now(timezone.utc),
    }
    db.assessment_sessions.insert_one(session_doc)

    if current_user.get("role") == "candidate":
        return {
            "session_id": session_doc["_id"],
            "personnel_id": personnel_id,
            "readiness_status": result.get("readiness_status"),
            "classification": result.get("classification"),
            "stress_probability": result["stress_probability"],
            "shap_attribution": result["shap_attribution"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Full technical response for commanders and medical officers
    return {
        "session_id": session_doc["_id"],
        "personnel_id": personnel_id,
        "features_used": features,
        **result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------
# Commander / Officer Dashboards (RBAC-protected)
# ---------------------------------------------------------
@router.get("/commander/roster")
def get_commander_roster(
    db=Depends(get_db),
    _current_user=Depends(require_role("commander", "medical_officer")),
):
    latest_sessions = list(
        db.assessment_sessions.find().sort("created_at", -1).limit(50)
    )
    seen = set()
    roster = []
    for s in latest_sessions:
        pid = s["personnel_id"]
        if pid in seen:
            continue
        seen.add(pid)
        roster.append({
            "candidate_id": pid,
            "readiness_tag": "Mandatory Rest Required" if s.get("classification") == "Critical Fatigue" else "Fit for Duty",
        })
    return {"total_evaluated": len(roster), "roster": roster}


@router.get("/welfare/triage")
def get_welfare_triage(
    db=Depends(get_db),
    _current_user=Depends(require_role("commander", "medical_officer")),
):
    critical_sessions = list(
        db.assessment_sessions.find({"classification": "Critical Fatigue"})
        .sort("created_at", -1)
        .limit(20)
    )
    pending = []
    for s in critical_sessions:
        shap_attr = s.get("shap_attribution") or []
        top_driver = shap_attr[0]["description"] if shap_attr else "High stress probability"
        pending.append({
            "personnel_id": s["personnel_id"],
            "risk_tier": "Critical",
            "primary_shap_driver": top_driver,
            "suggested_action": "Clinical rest order & psychological check-in.",
        })
    return {"pending_triages": pending}


@router.post("/welfare/interventions")
def log_welfare_intervention(
    data: InterventionPayload,
    db=Depends(get_db),
    _current_user=Depends(require_role("commander", "medical_officer")),
):
    doc = {
        "_id": gen_id(),
        "personnel_id": data.personnel_id,
        "action_type": data.action_type,
        "notes": data.notes,
        "created_at": datetime.now(timezone.utc),
    }
    db.welfare_interventions.insert_one(doc)
    return {"status": "success", "message": f"Intervention '{data.action_type}' recorded for {data.personnel_id}."}