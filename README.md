## Backend - Stress Detector for Army Personnel

Backend code for our project that monitors and detects stress levels for army personnel, built with FastAPI. Fully tested end-to-end, connected to a live MongoDB Atlas cluster. It processes physiological indicators (HRV), acoustic voice features, and behavioral tension metrics to calculate an overall stress index.

## Tech Stack
- **Framework:** FastAPI, Uvicorn
- **Data Processing:** OpenCV (`cv2`), MediaPipe, NumPy, SciPy, Librosa, FFmpeg
- **Database / Auth:** MongoDB (Atlas), PyMongo, python-jose (JWT), Passlib (Bcrypt)

### Structure
```
Backend/
├── main.py               # App entrypoint, router wiring
├── database.py           # MongoDB connection (reads MONGO_URI from .env)
├── models_db.py          # Collection index setup + ID generation
├── api/
│   ├── auth_api.py        # Register, login, JWT, RBAC
│   └── assessment_api.py  # Core evaluation + commander dashboard endpoints
└── pipelines/
    ├── video_processing.py  # Video/audio file I/O + ffmpeg demuxing
    └── pipeline_utils.py    # Signal processing algorithms (rPPG, HRV, blink, brow, voice) + scoring
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

### Data stored in MongoDB
- **personnel** — accounts (ID, name, hashed password, role, unit)
- **assessment_sessions** — every completed test: HR, HRV, blink rate, brow tension, voice pitch, duty/rest hours, final classification, and reasoning
- **welfare_interventions** — actions logged by commanders/medical officers on flagged cases

### Environment variables
Create a `.env` file inside `Backend/`:
```
JWT_SECRET_KEY=<any random string>
MONGO_URI=<your Atlas connection string>
MONGO_DB_NAME=stress_detector
GEMINI_API_KEY=<optional, for adaptive question generation>
```

## How to run it locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Shrestha-Ain/Stress_Detector.git
   cd Stress_Detector/Backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate.bat   # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create the `.env` file as described above.
5. Run the server:
   ```bash
   uvicorn main:app --reload --port 5000
   ```
   (Port 5000 used here due to a local port conflict on 8000 during development — use whichever port is free on your machine.)
6. Open `http://127.0.0.1:5000/docs` for interactive API testing.
