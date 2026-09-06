from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

router = APIRouter()

class UserRegister(BaseModel):
    personnel_id: str
    full_name: str
    password: str
    role: str = Field(..., example="candidate", description="candidate | commander | medical_officer")
    unit_id: str

@router.post("/register")
def register_personnel(user: UserRegister):
    return {
        "status": "success",
        "message": f"Account for {user.personnel_id} registered with role {user.role}."
    }

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    role = "medical_officer" if "doc" in form_data.username else ("commander" if "cmd" in form_data.username else "candidate")
    return {
        "access_token": f"mock_token_{form_data.username}",
        "token_type": "bearer",
        "role": role
    }

@router.get("/me")
def get_current_profile():
    return {"user_id": "CRPF-1042", "role": "medical_officer", "unit_id": "BN-07"}