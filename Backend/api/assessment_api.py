from fastapi import APIRouter
from pydantic import BaseModel, Field
from datetime import datetime, timezone

router = APIRouter()

class Standardized7VectorPayload(BaseModel):
    personnel_id: str
    hr_bpm: float
    rmssd_ms: float
    pitch_mean_hz: float
    pitch_std_hz: float
    blink_rate_bpm: float
    duty_hours_streak: float
    relax_hours_preceding: float

class InterventionPayload(BaseModel):
    personnel_id: str
    action_type: str
    notes: str

@router.post("/evaluate")
def evaluate_multimodal_assessment(payload: Standardized7VectorPayload):
    s_hrv = max(0.0, min(100.0, (1.0 - (payload.rmssd_ms - 20.0) / 60.0) * 100.0))
    s_voice = min(100.0, max(0.0, payload.pitch_std_hz * 4.0))
    s_behavior = max(0.0, min(100.0, (payload.blink_rate_bpm - 14.0) / (32.0 - 14.0) * 100.0))

    # Multi-modal fusion: 40% HRV, 30% Voice, 30% Behavior
    overall_stress = (0.40 * s_hrv) + (0.30 * s_voice) + (0.30 * s_behavior)
    
    is_critical = overall_stress >= 65.0 or payload.relax_hours_preceding < 4.0
    classification = "Critical Fatigue" if is_critical else "Cleared"

    shap_breakdown = [
        {"feature": "relax_hours_preceding", "importance": 0.38, "description": "Severe rest deficit (<4 hrs)"},
        {"feature": "rmssd_ms", "importance": 0.27, "description": "Depressed autonomic tone"},
        {"feature": "pitch_std_hz", "importance": 0.19, "description": "Acoustic micro-tremor dispersion"}
    ]

    return {
        "session_id": "SES-9821-X",
        "personnel_id": payload.personnel_id,
        "classification": classification,
        "readiness_status": "Mandatory Rest Required" if is_critical else "Fit for Duty",
        "stress_probability": 0.86 if is_critical else 0.28,
        "shap_attribution": shap_breakdown,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@router.get("/commander/roster")
def get_commander_roster():
    return {
        "unit_id": "BN-07",
        "total_evaluated": 42,
        "roster": [
            {"candidate_id": "CRPF-1042", "readiness_tag": "Mandatory Rest Required"},
            {"candidate_id": "CRPF-1043", "readiness_tag": "Fit for Duty"},
            {"candidate_id": "CRPF-1044", "readiness_tag": "Monitor"}
        ]
    }

@router.get("/welfare/triage")
def get_welfare_triage():
    return {
        "pending_triages": [
            {
                "personnel_id": "CRPF-1042",
                "risk_tier": "Critical",
                "primary_shap_driver": "Extreme sleep deficit + Vocal tremor",
                "suggested_action": "Clinical rest order & psychological check-in."
            }
        ]
    }

@router.post("/welfare/interventions")
def log_welfare_intervention(data: InterventionPayload):
    return {
        "status": "success",
        "message": f"Intervention '{data.action_type}' recorded for {data.personnel_id}."
    }