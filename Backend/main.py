from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth_api import router as auth_router
from api.opencv_apis import router as opencv_router
from api.assessment_api import router as model_router

app = FastAPI(
    title="Multi-Modal AI Personnel Stress & Welfare Monitoring API",
    version="2.0.0",
    description="SIH 186 Modular FastAPI Backend"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(auth_router, prefix="/api/auth", tags=["Auth & RBAC"])
app.include_router(opencv_router, prefix="/api/opencv", tags=["OpenCV & Telemetry"])
app.include_router(model_router, prefix="/api/assessment", tags=["ML Model & Dashboards"])

@app.get("/")
def root():
    return {"status": "online", "docs_url": "/docs"}