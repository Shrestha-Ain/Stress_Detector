"""
Pure processing module — NOT an API router. No HTTP endpoints live here;
assessment_api.py imports these two functions directly and calls them
inside /api/assessment/full-evaluate. Kept separate from pipeline_utils.py
because these two functions handle FILE I/O (reading the uploaded video,
running ffmpeg as a subprocess), while pipeline_utils.py stays pure
signal-processing with no file/process dependencies.

Adapted from the ML teammate's main_pipeline.py: her version runs live
against cv2.VideoCapture(0) with cv2.imshow + waitKey(ENTER) — a desktop
demo script. This version does the same landmark math (forehead rPPG,
EAR blink counting, brow tension, head jitter) but reads frames from an
uploaded video file instead of a live camera, with no display/keyboard
wait, so it can run headless inside a web request.
"""
import os
import cv2
import mediapipe as mp
import numpy as np
import librosa
import subprocess

from pipelines.pipeline_utils import (
    eye_aspect_ratio, pos_algorithm, bandpass_filter, estimate_hr_and_hrv,
    BlinkCounter, extract_voice_stress_features,
)

# Same landmark indices as the ML teammate's main_pipeline.py
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_BROW_INNER = 107
RIGHT_BROW_INNER = 336
NOSE_TIP = 1


def get_pts(landmarks, indices, w, h):
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])


def run_opencv_processing(temp_path: str):
    """
    Reads the uploaded video, runs MediaPipe FaceMesh per frame, and extracts:
      - hr_bpm, rmssd_ms   (forehead rPPG via pos_algorithm/bandpass_filter/estimate_hr_and_hrv)
      - blink_rate         (EAR-based blink counting)
      - brow_ratio         (inner-eyebrow distance / face width — tension proxy)
      - head_jitter        (std of nose-tip position — movement/restlessness proxy)
    """
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise ValueError("Could not read uploaded video file.")

    rgb_means = []
    timestamps = []
    brow_ratios = []
    head_positions = []
    blink_counter = BlinkCounter()
    frame_idx = 0

    use_mediapipe = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
    face_mesh_ctx = None

    if use_mediapipe:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh_ctx = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6
        )

    try:
        if face_mesh_ctx:
            face_mesh_ctx.__enter__()

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx % 2 != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            if w > 640 and h > 0:
                scale = 640 / w
                w = 640
                h = max(1, int(h * scale))
                frame = cv2.resize(frame, (w, h))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_time = frame_idx / 30.0

            if face_mesh_ctx:
                results = face_mesh_ctx.process(rgb_frame)
                if results and results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    xs = [lm.x * w for lm in landmarks]
                    ys = [lm.y * h for lm in landmarks]
                    x1, x2 = int(min(xs) + 0.30 * (max(xs) - min(xs))), int(
                        min(xs) + 0.70 * (max(xs) - min(xs))
                    )
                    y1, y2 = int(min(ys) + 0.08 * (max(ys) - min(ys))), int(
                        min(ys) + 0.28 * (max(ys) - min(ys))
                    )
                    forehead = rgb_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if forehead.size > 0:
                        rgb_means.append(forehead.reshape(-1, 3).mean(axis=0))
                        timestamps.append(current_time)

                    # Blink detection
                    right_pts = get_pts(landmarks, RIGHT_EYE, w, h)
                    left_pts = get_pts(landmarks, LEFT_EYE, w, h)
                    ear = (eye_aspect_ratio(right_pts) + eye_aspect_ratio(left_pts)) / 2.0
                    blink_counter.update(ear)

                    # Brow tension: inner-eyebrow distance normalized by face width
                    face_width = max(xs) - min(xs)
                    brow_dist = np.linalg.norm(
                        np.array([landmarks[LEFT_BROW_INNER].x * w, landmarks[LEFT_BROW_INNER].y * h]) -
                        np.array([landmarks[RIGHT_BROW_INNER].x * w, landmarks[RIGHT_BROW_INNER].y * h])
                    )
                    brow_ratios.append(brow_dist / max(face_width, 1.0))

                    # Head jitter: nose-tip position over time (std = restlessness)
                    head_positions.append([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
                else:
                    forehead = rgb_frame[int(h * 0.3):int(h * 0.5), int(w * 0.4):int(w * 0.6)]
                    if forehead.size > 0:
                        rgb_means.append(forehead.reshape(-1, 3).mean(axis=0))
                        timestamps.append(current_time)
            else:
                forehead = rgb_frame[int(h * 0.3):int(h * 0.5), int(w * 0.4):int(w * 0.6)]
                if forehead.size > 0:
                    rgb_means.append(forehead.reshape(-1, 3).mean(axis=0))
                    timestamps.append(current_time)

            frame_idx += 1
    finally:
        if face_mesh_ctx:
            face_mesh_ctx.__exit__(None, None, None)
        cap.release()

    total_blinks = blink_counter.finalize()

    actual_fps = (
        len(timestamps) / (timestamps[-1] - timestamps[0])
        if len(timestamps) > 1
        else 30.0
    )
    hr_bpm, rmssd_ms = 72.0, 45.0

    if len(rgb_means) > 15:
        try:
            pulse = pos_algorithm(np.array(rgb_means), actual_fps)
            filtered = bandpass_filter(pulse, actual_fps)
            hrv = estimate_hr_and_hrv(filtered, actual_fps)
            hr_bpm = hrv.get("hr_bpm") or 72.0
            rmssd_ms = hrv.get("rmssd_ms") or 45.0
        except Exception:
            pass

    duration_sec = frame_idx / actual_fps if actual_fps > 0 else 1.0
    blink_rate = (total_blinks / duration_sec) * 60.0 if duration_sec > 0 else 0.0
    brow_ratio = float(np.mean(brow_ratios)) if brow_ratios else 0.20
    head_jitter = float(np.std(head_positions)) if head_positions else 1.0

    return {
        "frame_count": frame_idx,
        "hr_bpm": round(float(hr_bpm), 2),
        "rmssd_ms": round(float(rmssd_ms), 2),
        "blink_rate": round(float(blink_rate), 2),
        "brow_ratio": round(brow_ratio, 4),
        "head_jitter": round(head_jitter, 4),
    }


def extract_audio_track(video_path: str, target_sr: int = 22050):
    """Demuxes the audio track out of the browser-recorded video via ffmpeg.
    Returns a temp .wav path, or None if there's no audio track / ffmpeg
    isn't available."""
    wav_path = f"{video_path}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(target_sr), "-ac", "1",
        wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0 or not os.path.exists(wav_path):
        return None
    return wav_path


def run_audio_processing_from_video(video_path: str):
    """Extracts the audio track from the video and scores it via
    pipeline_utils.extract_voice_stress_features. Falls back to neutral
    values (never raises) if there's no audio track or ffmpeg fails."""
    wav_path = extract_audio_track(video_path)
    if wav_path is None:
        return {"pitch_mean_hz": 0.0, "pitch_std_hz": 0.0, "vocal_stress_subscore": 50.0}
    try:
        audio_data, sr = librosa.load(wav_path, sr=22050, mono=True)
        return extract_voice_stress_features(audio_data, sr=sr)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)