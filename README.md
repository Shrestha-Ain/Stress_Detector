# Stress Detector for Army Personnel

Backend code for our project that monitors and detects stress levels for army personnel, built with FastAPI.It processes physiological indicators (HRV), acoustic voice features, and behavioral tension metrics to calculate an overall stress index.

## Core Features
* **Multi-Modal Fusion Scoring**: Combines physiological baseline metrics (40%), voice acoustic stress (30%), and behavioral tension/blink rates (30%).
* **FastAPI Backend**: Modular route architecture handling authentication (`auth_apis.py`), biometric video processing (`opencv_apis.py`), and stress triage evaluation (`assessment_apis.py`).
* **Dynamic Evaluation**: Computes real-time clinical status and risk classifications from standardized 7-vector payloads.

## Tech Stack
* **Framework**: FastAPI, Uvicorn
* **Computer Vision & Signal Processing**: OpenCV, MediaPipe, NumPy
* **Audio Analysis**: Sounddevice, Librosa
* **Validation**: Pydantic

## How to run it locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Shrestha-Ain/Stress_Detector.git
   cd Stress_Detector


