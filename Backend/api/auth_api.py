from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, Field
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
import os

from database import get_db
from models_db import gen_id

router = APIRouter()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class UserRegister(BaseModel):
    personnel_id: str
    full_name: str
    password: str
    role: str = Field(..., examples=["candidate"], description="candidate | commander | medical_officer")
    unit_id: str


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        personnel_id: str = payload.get("sub")
        if personnel_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.personnel.find_one({"personnel_id": personnel_id})
    if user is None:
        raise credentials_exception
    return user


def require_role(*allowed_roles: str):
    """Dependency factory for RBAC: Depends(require_role('commander', 'medical_officer'))"""

    def role_checker(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.get('role')}' is not permitted to access this resource.",
            )
        return user

    return role_checker


@router.post("/register")
def register_personnel(user: UserRegister, db=Depends(get_db)):
    existing = db.personnel.find_one({"personnel_id": user.personnel_id})
    if existing:
        raise HTTPException(status_code=400, detail="Personnel ID already registered.")

    doc = {
        "_id": gen_id(),
        "personnel_id": user.personnel_id,
        "full_name": user.full_name,
        "hashed_password": pwd_context.hash(user.password),
        "role": user.role,
        "unit_id": user.unit_id,
        "baseline_hr_bpm": None,
        "baseline_pitch_hz": None,
        "shift_type": None,
        "duty_hours_streak": None,
        "relax_hours_preceding": None,
        "created_at": datetime.now(timezone.utc),
    }
    db.personnel.insert_one(doc)
    return {"status": "success", "message": f"Account for {user.personnel_id} registered with role {user.role}."}


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    user = db.personnel.find_one({"personnel_id": form_data.username})
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect personnel ID or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user["personnel_id"], "role": user["role"]})
    return {"access_token": access_token, "token_type": "bearer", "role": user["role"]}


@router.get("/me")
def get_current_profile(current_user: dict = Depends(get_current_user)):
    return {
        "user_id": current_user["personnel_id"],
        "full_name": current_user["full_name"],
        "role": current_user["role"],
        "unit_id": current_user.get("unit_id"),
    }