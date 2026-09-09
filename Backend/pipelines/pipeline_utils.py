import numpy as np
from scipy import signal as sps
import librosa


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

    # Welch PSD dominant frequency search
    freqs, psd = sps.welch(pulse_signal - np.mean(pulse_signal), fs=fps, nperseg=min(n, int(fps * 6)))
    valid = (freqs >= 0.75) & (freqs <= 3.0)
    if not np.any(valid):
        return empty

    peak_freq = freqs[valid][np.argmax(psd[valid])]
    hr_bpm = peak_freq * 60.0

    # IBI estimation via peak picking
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
        "rmssd_ms": round(float(min(rmssd_ms, 150.0)), 1)
    }


# ---------------------------------------------------------
# 2. Eye Aspect Ratio & Brow Tension
# ---------------------------------------------------------
def eye_aspect_ratio(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4) + 1e-6)


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

    # Normalize audio
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)

    # 1. Fundamental Frequency (F0 / Pitch) via PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )
    valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    if len(valid_f0) > 5:
        pitch_mean = float(np.mean(valid_f0))
        pitch_std = float(np.std(valid_f0))
    else:
        pitch_mean, pitch_std = 120.0, 5.0

    # 2. Spectral Centroid & Energy
    spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    mean_spectral_centroid = float(np.mean(spec_cent))

    # Vocal Stress Scoring: higher pitch variance and high spectral brightness -> tension
    s_pitch_var = np.clip((pitch_std - 15.0) / (45.0 - 15.0) * 100.0, 0, 100)
    s_spectral = np.clip((mean_spectral_centroid - 1200.0) / (2800.0 - 1200.0) * 100.0, 0, 100)
    vocal_subscore = 0.60 * s_pitch_var + 0.40 * s_spectral

    return {
        "pitch_mean_hz": round(pitch_mean, 1),
        "pitch_std_hz": round(pitch_std, 1),
        "vocal_stress_subscore": round(float(vocal_subscore), 1)
    }