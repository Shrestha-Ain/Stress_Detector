import numpy as np
from scipy import signal as sps
import librosa
import joblib
import os


# ---------------------------------------------------------
# 1. rPPG & HRV Processing
# ---------------------------------------------------------
def pos_algorithm(rgb_signal, fps):
    rgb_signal = np.asarray(rgb_signal, dtype=np.float64)
    n = rgb_signal.shape[0]
    window_len = int(fps * 1.6)
    if n < window_len:
        return np.zeros(n)

    pulse = np.zeros(n)
    weights = np.zeros(n)

    for start in range(0, n - window_len + 1):
        window = rgb_signal[start:start + window_len]
        mean_rgb = np.mean(window, axis=0)
        mean_rgb[mean_rgb == 0] = 1e-6
        normalized = window / mean_rgb

        Xs = 3 * normalized[:, 0] - 2 * normalized[:, 1]
        Ys = 1.5 * normalized[:, 0] + normalized[:, 1] - 1.5 * normalized[:, 2]

        std_x = np.std(Xs)
        std_y = np.std(Ys)
        alpha = std_x / std_y if std_y > 1e-8 else 0

        S = Xs - alpha * Ys
        pulse[start:start + window_len] += (S - np.mean(S))
        weights[start:start + window_len] += 1.0

    return np.divide(pulse, weights, out=np.zeros_like(pulse), where=weights > 0)


def bandpass_filter(sig, fps, low_hz=0.75, high_hz=3.0, order=3):
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    if n < 15:
        return sig - np.mean(sig)

    nyq = fps / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    b, a = sps.butter(order, [low, high], btype="band")
    padlen = min(n - 1, 3 * max(len(a), len(b)))
    return sps.filtfilt(b, a, sig, padlen=padlen)


def estimate_hr_and_hrv(pulse_signal, fps):
    pulse_signal = np.asarray(pulse_signal, dtype=np.float64)
    n = len(pulse_signal)
    empty = {"hr_bpm": None, "rmssd_ms": None}
    if n < int(fps * 3) or np.std(pulse_signal) < 1e-6:
        return empty

    freqs, psd = sps.welch(pulse_signal - np.mean(pulse_signal), fs=fps, nperseg=min(n, int(fps * 6)))
    valid = (freqs >= 0.75) & (freqs <= 3.0)
    if not np.any(valid):
        return empty

    peak_freq = freqs[valid][np.argmax(psd[valid])]
    hr_bpm = peak_freq * 60.0

    expected_dist = int(fps / peak_freq)
    peaks, _ = sps.find_peaks(pulse_signal, distance=max(int(expected_dist * 0.75), 1), prominence=0.30 * np.std(pulse_signal))

    if len(peaks) < 3:
        return {"hr_bpm": round(float(hr_bpm), 1), "rmssd_ms": None}

    ibi_ms = (np.diff(peaks) / fps) * 1000.0
    med = np.median(ibi_ms)
    clean_ibi = ibi_ms[(ibi_ms >= 0.80 * med) & (ibi_ms <= 1.20 * med)]

    if len(clean_ibi) < 2:
        return {"hr_bpm": round(float(hr_bpm), 1), "rmssd_ms": None}

    rmssd_ms = np.sqrt(np.mean(np.diff(clean_ibi) ** 2))
    return {
        "hr_bpm": round(float(hr_bpm), 1),
        "rmssd_ms": round(float(min(rmssd_ms, 150.0)), 1),
    }


# ---------------------------------------------------------
# 2. Eye Aspect Ratio & Blink Engine
# ---------------------------------------------------------
def eye_aspect_ratio(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4) + 1e-6)


EAR_BLINK_THRESHOLD = 0.21
EAR_CONSEC_FRAMES = 2


class BlinkCounter:
    """
    Tracks EAR across consecutive frames and counts a blink whenever EAR dips
    below threshold for at least `consec_frames` in a row, then recovers.
    Call .update(ear) per processed frame, .finalize() once at the end to
    flush a blink that was still in progress when the video ended.
    """

    def __init__(self, threshold: float = EAR_BLINK_THRESHOLD, consec_frames: int = EAR_CONSEC_FRAMES):
        self.threshold = threshold
        self.consec_frames = consec_frames
        self._below_count = 0
        self.blink_count = 0

    def update(self, ear_value: float):
        if ear_value < self.threshold:
            self._below_count += 1
        else:
            if self._below_count >= self.consec_frames:
                self.blink_count += 1
            self._below_count = 0

    def finalize(self) -> int:
        if self._below_count >= self.consec_frames:
            self.blink_count += 1
            self._below_count = 0
        return self.blink_count


# ---------------------------------------------------------
# 3. Voice Feature Extraction (Acoustic Stress Biomarkers)
# ---------------------------------------------------------
def extract_voice_stress_features(audio_data, sr=22050):
    """
    Extracts fundamental frequency (F0), jitter approximation,
    spectral centroid, and energy variance using librosa.
    """
    if len(audio_data) < sr * 1:
        return {"pitch_mean_hz": 0.0, "pitch_std_hz": 0.0, "vocal_stress_subscore": 50.0}

    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_data, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    if len(valid_f0) > 5:
        pitch_mean = float(np.mean(valid_f0))
        pitch_std = float(np.std(valid_f0))
    else:
        pitch_mean, pitch_std = 120.0, 5.0

    spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    mean_spectral_centroid = float(np.mean(spec_cent))

    s_pitch_var = np.clip((pitch_std - 15.0) / (45.0 - 15.0) * 100.0, 0, 100)
    s_spectral = np.clip((mean_spectral_centroid - 1200.0) / (2800.0 - 1200.0) * 100.0, 0, 100)
    vocal_subscore = 0.60 * s_pitch_var + 0.40 * s_spectral

    return {
        "pitch_mean_hz": round(pitch_mean, 1),
        "pitch_std_hz": round(pitch_std, 1),
        "vocal_stress_subscore": round(float(vocal_subscore), 1),
    }


# ---------------------------------------------------------
# 4. Multi-Modal ML Classification (SVM + SHAP, with fallback)
# ---------------------------------------------------------

_MODEL_PATH = os.getenv("STRESS_MODEL_PATH", "models/stress_svm.joblib")
_SCALER_PATH = os.getenv("STRESS_SCALER_PATH", "models/stress_scaler.joblib")
_EXPLAINER_PATH = os.getenv("STRESS_EXPLAINER_PATH", "models/stress_explainer.joblib")

_FEATURE_ORDER = [
    "hr_bpm", "rmssd_ms", "pitch_mean_hz", "pitch_std_hz",
    "blink_rate_bpm", "brow_ratio", "duty_hours_streak", "relax_hours_preceding",
]

_model = None
_scaler = None
_explainer = None
_model_load_attempted = False


def _try_load_model():
    global _model, _scaler, _explainer, _model_load_attempted
    if _model_load_attempted:
        return
    _model_load_attempted = True
    try:
        _model = joblib.load(_MODEL_PATH)
        _scaler = joblib.load(_SCALER_PATH)
        if os.path.exists(_EXPLAINER_PATH):
            _explainer = joblib.load(_EXPLAINER_PATH)
    except Exception:
        _model, _scaler, _explainer = None, None, None


def _heuristic_score(features: dict) -> dict:
    """
    Deterministic weighted fallback — used until a trained SVM exists.
    Matches the ML teammate's actual fuse_multimodal_scores() formula from
    main_pipeline.py: 40% HRV + 30% voice + 30% behavior, where behavior
    itself is 60% blink-rate + 40% brow tension.
    """
    s_hrv = max(0.0, min(100.0, (1.0 - (features["rmssd_ms"] - 20.0) / 60.0) * 100.0))
    s_voice = min(100.0, max(0.0, features["pitch_std_hz"] * 4.0))

    s_blink = max(0.0, min(100.0, (features["blink_rate_bpm"] - 14.0) / (32.0 - 14.0) * 100.0))
    brow_ratio = features.get("brow_ratio", 0.20)  # lower ratio = more furrowed/tense
    s_brow = max(0.0, min(100.0, (0.22 - brow_ratio) / (0.22 - 0.14) * 100.0))
    s_behavior = 0.60 * s_blink + 0.40 * s_brow

    rest_deficit = features["relax_hours_preceding"] < 4.0

    overall = (0.40 * s_hrv) + (0.30 * s_voice) + (0.30 * s_behavior)
    probability = min(0.97, max(overall, 65.0 if rest_deficit else 0.0) / 100.0)
    is_critical = probability >= 0.65 or rest_deficit

    shap_breakdown = sorted(
        [
            {"feature": "relax_hours_preceding", "importance": round(0.38 if rest_deficit else 0.10, 2),
            "description": "Rest deficit in preceding 48h"},
            {"feature": "rmssd_ms", "importance": round(s_hrv / 100 * 0.4, 2),
            "description": "Depressed autonomic recovery (HRV)"},
            {"feature": "pitch_std_hz", "importance": round(s_voice / 100 * 0.3, 2),
            "description": "Vocal micro-tremor dispersion"},
            {"feature": "blink_rate_bpm", "importance": round(s_blink / 100 * 0.3 * 0.6, 2),
            "description": "Elevated blink rate"},
            {"feature": "brow_ratio", "importance": round(s_brow / 100 * 0.3 * 0.4, 2),
            "description": "Brow furrowing / facial tension"},
        ],
        key=lambda f: f["importance"],
        reverse=True,
    )[:3]

    return {
        "classification": "Critical Fatigue" if is_critical else "Cleared",
        "readiness_status": "Mandatory Rest Required" if is_critical else "Fit for Duty",
        "stress_probability": round(probability, 2),
        "shap_attribution": shap_breakdown,
        "scoring_method": "heuristic_fallback",
    }


def score_stress(features: dict) -> dict:
    """
    Scores a 7-feature vector (see _FEATURE_ORDER for required keys).
    Uses the trained RBF-SVM + SHAP explainer if present in ./models/,
    otherwise falls back to _heuristic_score so the pipeline never breaks.
    """
    _try_load_model()

    if _model is not None and _scaler is not None:
        try:
            x = [[features[k] for k in _FEATURE_ORDER]]
            x_scaled = _scaler.transform(x)
            proba = _model.predict_proba(x_scaled)[0]
            classes = list(_model.classes_)
            critical_idx = classes.index(1) if 1 in classes else int(np.argmax(proba))
            probability = float(proba[critical_idx])
            is_critical = probability >= 0.5

            shap_breakdown = []
            if _explainer is not None:
                try:
                    shap_values = _explainer.shap_values(x_scaled)
                    contributions = shap_values[critical_idx][0] if isinstance(shap_values, list) else shap_values[0]
                    ranked = sorted(zip(_FEATURE_ORDER, contributions), key=lambda p: abs(p[1]), reverse=True)[:3]
                    shap_breakdown = [
                        {"feature": name, "importance": round(float(val), 3),
                        "description": f"SHAP contribution for {name}"}
                        for name, val in ranked
                    ]
                except Exception:
                    shap_breakdown = []

            return {
                "classification": "Critical Fatigue" if is_critical else "Cleared",
                "readiness_status": "Mandatory Rest Required" if is_critical else "Fit for Duty",
                "stress_probability": round(probability, 2),
                "shap_attribution": shap_breakdown,
                "scoring_method": "svm_model",
            }
        except Exception:
            pass

    return _heuristic_score(features)


# ---------------------------------------------------------
# 5. Generative AI Context Engine (Gemini 2.5 Flash + fallback)
# ---------------------------------------------------------
_FALLBACK_QUESTIONS = [
    "On a scale of 1 to 10, how would you rate your energy level right now?",
    "Have you had any trouble sleeping in the past two nights?",
    "Do you feel any unusual tension or discomfort at this moment?",
]


def generate_adaptive_question(hr_bpm: float, rmssd_ms: float, baseline_hr_bpm: float = None) -> dict:
    """
    Calls Gemini 2.5 Flash to generate a targeted follow-up question based on
    Stage-1 HR/HRV anomalies. Uses the `google-genai` SDK (google.genai) —
    matching the ML teammate's actual working implementation in
    main_pipeline.py's generate_adaptive_prompt(), NOT the older
    google-generativeai package. Falls back to a fixed question with zero
    latency if GEMINI_API_KEY is missing, the request fails, or times out.

    Not currently wired to a live endpoint (no mid-recording pause in the
    single-upload flow) — kept here ready to use if that feature comes back.
    """
    import random
    import json

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"question": _FALLBACK_QUESTIONS[0], "source": "fallback"}

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        topics = [
            "recent sleep quality and physical recovery",
            "current cognitive workload and focus",
            "recent feelings of being rushed or overwhelmed",
            "ability to disconnect and relax after a long day",
            "recent changes in appetite or daily routine",
        ]
        chosen_topic = random.choice(topics)

        prompt = f"""
        You are an objective clinical screening assistant.
        The candidate has completed their baseline scan:
        - Resting HR: {hr_bpm} BPM
        - HRV (RMSSD): {rmssd_ms} ms (Lower = higher strain)

        Generate exactly ONE targeted question to assess their {chosen_topic}.
        Keep the question conversational, under 20 words, and do not use standard greetings.

        Return JSON with this exact schema:
        {{
           "question_text": "...",
           "focus_area": "{chosen_topic}"
        }}
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8,
            ),
        )
        parsed = json.loads(response.text)
        question = parsed.get("question_text", "").strip()
        if not question:
            raise ValueError("Empty question_text from Gemini")
        return {"question": question, "source": "gemini"}
    except Exception:
        return {"question": _FALLBACK_QUESTIONS[1], "source": "fallback"}