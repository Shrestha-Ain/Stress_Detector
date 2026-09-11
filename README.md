## Backend

FastAPI backend for Stress Detector for Army Personnel

Backend code for our project that monitors and detects stress levels for army personnel, built with FastAPI.It processes physiological indicators (HRV), acoustic voice features, and behavioral tension metrics to calculate an overall stress index.

## Tech Stack
- **Framework:** FastAPI, Uvicorn
- **Data Processing:** OpenCV (`cv2`), NumPy, SciPy
- **Database / Auth:** SQLite / MongoDB, PyJWT, Bcrypt

### Structure
```
Backend/
├── main.py              # App entrypoint, router wiring
├── database.py          # DB connection (in-memory stand-in until MongoDB Atlas is connected)
├── models_db.py         # Collection index setup + ID generation
├── api/
│   ├── auth_api.py       # Register, login, JWT, RBAC
│   └── assessment_api.py # Core evaluation + commander dashboard endpoints
└── pipelines/
    ├── video_processing.py  # Video/audio file I/O + ffmpeg demuxing
    └── pipeline_utils.py    # Signal processing algorithms (rPPG, HRV, blink, voice) + scoring
```

### Endpoints
| Endpoint | Purpose |
|---|---|
| `POST /api/auth/register` | Create an account |
| `POST /api/auth/login` | Get a JWT access token |
| `GET /api/auth/me` | Get current user's profile |
| `POST /api/assessment/full-evaluate` | Submit video + duty/sleep hours → get stress classification |
| `GET /api/assessment/commander/roster` | Latest status per person (commander/medical_officer only) |
| `GET /api/assessment/welfare/triage` | Critical cases + reasoning (commander/medical_officer only) |
| `POST /api/assessment/welfare/interventions` | Log an action taken on a flagged case |

### Environment variables
```
JWT_SECRET_KEY=<random string>
MONGO_URI=<Atlas connection string, once available>
GEMINI_API_KEY=<optional, for adaptive question generation>
```

## How to run it locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Shrestha-Ain/Stress_Detector.git
   cd Stress_Detector
