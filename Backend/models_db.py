"""
No ORM classes here — Mongo is schemaless, so this file just documents the
document shapes each collection holds, plus index setup and an id helper.
Collections used: personnel, assessment_sessions, welfare_interventions.

personnel document:
{
    "_id": <uuid str>,
    "personnel_id": str,           # unique, natural key used everywhere else
    "full_name": str,
    "hashed_password": str,
    "role": str,                   # candidate | commander | medical_officer
    "unit_id": str,
    "baseline_hr_bpm": float | None,
    "baseline_pitch_hz": float | None,
    "created_at": datetime,
}
# Note: duty_hours_streak / relax_hours_preceding are NOT stored here —
# the candidate sends them fresh with every /full-evaluate call instead,
# so they only ever live on the assessment_sessions record below.

assessment_sessions document:
{
    "_id": <uuid str>,
    "personnel_id": str,
    "duty_hours_streak": float,
    "relax_hours_preceding": float,
    "hr_bpm": float,
    "rmssd_ms": float,
    "blink_rate_bpm": float,
    "brow_ratio": float,
    "head_jitter": float,          # logged for now, not yet used in scoring
    "pitch_mean_hz": float,
    "pitch_std_hz": float,
    "stress_probability": float,
    "classification": str,         # "Cleared" | "Critical Fatigue"
    "shap_attribution": list[dict],
    "created_at": datetime,
}

welfare_interventions document:
{
    "_id": <uuid str>,
    "personnel_id": str,
    "action_type": str,
    "notes": str,
    "created_at": datetime,
}
"""
from pymongo import ASCENDING, DESCENDING
from database import db
import uuid


def gen_id() -> str:
    return str(uuid.uuid4())


def ensure_indexes():
    """Call once at startup (see main.py). Safe to call repeatedly — Mongo
    no-ops if the index already exists with the same spec."""
    db.personnel.create_index([("personnel_id", ASCENDING)], unique=True)
    db.assessment_sessions.create_index([("personnel_id", ASCENDING), ("created_at", DESCENDING)])
    db.assessment_sessions.create_index([("classification", ASCENDING), ("created_at", DESCENDING)])
    db.welfare_interventions.create_index([("personnel_id", ASCENDING)])