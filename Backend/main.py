from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models_db import ensure_indexes
from api.auth_api import router as auth_router
from api.assessment_api import router as assessment_router

# Sets up Mongo indexes on startup (unique personnel_id, query indexes on
# assessment_sessions). Safe to call every run.
ensure_indexes()

app = FastAPI(
    title="Multi-Modal AI Personnel Stress & Welfare Monitoring API",
    version="4.0.0",
    description="SIH 186 Modular FastAPI Backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router, prefix="/api/auth", tags=["Auth & RBAC"])
app.include_router(assessment_router, prefix="/api/assessment", tags=["Assessment & Dashboards"])


@app.get("/")
def root():
    return {"status": "online", "docs_url": "/docs"}