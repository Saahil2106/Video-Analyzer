import cv2
import numpy as np
from pathlib import Path
from scipy.fft import dct
from cinema_analyzer import analyze_video_cinema
from json_utils import to_json_safe

FEATURES = [
    "mean_flow", "max_flow", "std_flow", "flow_variance",
    "frame_diff_mean", "frame_diff_std", "freq_ratio",
    "freeze_ratio", "accel_score", "periodicity_score",
    "rotation_score", "flow_gradient"
]

def extract_features(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_gray = None
    flow_magnitudes = []
    frame_diffs = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_magnitudes.append(float(mag.mean()))
            diff = cv2.absdiff(prev_gray, gray)
            frame_diffs.append(float(diff.mean()))

        prev_gray = gray
        frame_count += 1

    cap.release()

    if len(flow_magnitudes) < 5:
        return None

    flow_arr = np.array(flow_magnitudes)
    diff_arr = np.array(frame_diffs)

    dct_coeffs = dct(diff_arr, norm="ortho")
    low_energy = float(np.sum(dct_coeffs[:5]**2))
    high_energy = float(np.sum(dct_coeffs[5:]**2))
    freq_ratio = high_energy / (low_energy + 1e-8)

    freeze_ratio = float(np.sum(flow_arr < 0.3) / len(flow_arr))
    half = len(flow_arr) // 2
    accel_score = float(flow_arr[half:].mean() - flow_arr[:half].mean())

    flow_norm = flow_arr - flow_arr.mean()
    autocorr = np.correlate(flow_norm, flow_norm, mode="full")
    autocorr = autocorr[len(autocorr)//2:]
    autocorr_norm = autocorr / (autocorr[0] + 1e-8)
    periodicity_score = float(autocorr_norm[5:20].max()) \
        if len(autocorr) > 20 else 0.0
    rotation_score = float(flow_arr.std() / (flow_arr.mean() + 1e-8))
    flow_gradient = float(np.abs(np.diff(flow_arr)).mean())

    duration = round(frame_count / fps, 2)

    # Temporal segments — sliding window
    win = max(1, int(fps * 1.5))
    segs = []
    step = max(1, win // 2)
    for i in range(0, max(1, len(flow_magnitudes) - win + 1), step):
        chunk = flow_arr[i:i + win]
        if len(chunk) == 0:
            continue
        t_start = round(i / fps, 2)
        t_end = round((i + len(chunk)) / fps, 2)
        segs.append({
            "t_start": t_start,
            "t_end": t_end,
            "mean_flow": round(float(chunk.mean()), 3),
            "freeze": bool(float(np.sum(chunk < 0.3) / len(chunk)) > 0.4),
            "burst": bool(float(chunk.max()) > 3.0),
        })

    result = {
        "num_frames": frame_count,
        "fps": round(fps, 2),
        "duration_sec": duration,
        "mean_flow": round(float(flow_arr.mean()), 4),
        "max_flow": round(float(flow_arr.max()), 4),
        "std_flow": round(float(flow_arr.std()), 4),
        "flow_variance": round(float(np.var(flow_arr)), 4),
        "frame_diff_mean": round(float(diff_arr.mean()), 4),
        "frame_diff_std": round(float(diff_arr.std()), 4),
        "freq_ratio": round(freq_ratio, 4),
        "freeze_ratio": round(freeze_ratio, 4),
        "accel_score": round(accel_score, 4),
        "periodicity_score": round(periodicity_score, 4),
        "rotation_score": round(rotation_score, 4),
        "flow_gradient": round(flow_gradient, 4),
        "segments": segs,
    }
    return to_json_safe(result)

def build_metadata(video_path: str, predictions: dict,
features: dict, tracking_data: dict = None) -> dict:
    path = Path(video_path)
    dominant = max(predictions, key=predictions.get)

    intensity = (
        "high" if features["mean_flow"] > 2.0 else
        "medium" if features["mean_flow"] > 1.0 else
        "low"
    )

    effects_list = [
        {"effect": e, "confidence": round(c * 100, 1)}
        for e, c in predictions.items() if c > 0.08
    ]

    timeline = []
    for seg in features.get("segments", []):
        if seg["freeze"]:
            label = "freeze_moment"
        elif seg["burst"]:
            label = "rapid_motion"
        elif seg["mean_flow"] > 2.0:
            label = "speed_ramp"
        elif seg["mean_flow"] < 0.5:
            label = "smooth_motion"
        else:
            label = dominant
        timeline.append({
            "start": seg["t_start"],
            "end": seg["t_end"],
            "effect": label,
        })

    summary_parts = []
    if features["freeze_ratio"] > 0.3:
        summary_parts.append("notable freeze moments")
    if features["accel_score"] > 0.2:
        summary_parts.append("accelerating motion toward the end")
    if features["accel_score"] < -0.2:
        summary_parts.append("decelerating motion")
    if features["periodicity_score"] > 0.7:
        summary_parts.append("rhythmic/repetitive motion")
    if features["rotation_score"] > 1.0:
        summary_parts.append("spin or rotation effects")
    summary = f"{intensity.capitalize()} motion intensity clip"
    summary += (" with " + ", ".join(summary_parts) + "."
    if summary_parts else ".")

    # Build detections_by_frame from tracking data if available
    detections_by_frame = None
    if tracking_data:
        detections_by_frame = {}
        for shot in tracking_data.get("shots", []):
            for frame in shot.get("frames", []):
                detections_by_frame[frame["frame"]] = frame["detections"]

    print(f" Running cinema analysis on {path.name}...")
    cinema = analyze_video_cinema(
        video_path,
        sample_rate=15,
        detections_by_frame=detections_by_frame,
    )

    # ── Per-segment optical flow (sensor tier) ──────────────────
    per_segment_flow = [
        {
            "t_start": seg["t_start"],
            "t_end": seg["t_end"],
            "mean_flow": seg["mean_flow"],
            "freeze": seg["freeze"],
            "burst": seg["burst"],
        }
        for seg in features.get("segments", [])
    ]

    result = {
        # ── BASIC METADATA ──────────────────────────────────────
        # File identity, playback info, top-level effect summary.
        # Suitable for catalogue indexing and quick preview.
        "basic_metadata": {
            "file": path.name,
            "duration_seconds": features["duration_sec"],
            "fps": features["fps"],
            "total_frames": features["num_frames"],
            "motion_intensity": intensity,
            "dominant_effect": dominant,
            "effects_detected": effects_list,
            "summary": summary,
        },

        # ── ADVANCED METADATA ───────────────────────────────────
        # Effect timeline, cinematography analysis, subject tracking.
        # Requires model inference + cinema analysis pipeline.
        "advanced_metadata": {
            "timeline": timeline,
            "cinematography": {
                "clip_summary": cinema["summary"] if cinema else {},
                "segments": cinema["segments"] if cinema else [],
            },
            "subject_tracking": (
                tracking_data.get("clip_summary", {})
                if tracking_data else None
            ),
        },

        # ── SENSOR METADATA ─────────────────────────────────────
        # Raw optical-flow scalars and per-window signal data.
        # Low-level numbers produced directly by the CV pipeline.
        "sensor_metadata": {
            "raw_features": {
                k: v for k, v in features.items()
                if k != "segments"
            },
            "per_segment_flow": per_segment_flow,
        },
    }

    return to_json_safe(result)