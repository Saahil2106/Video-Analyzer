"""
feature_extractor.py
────────────────────
Standalone feature extraction pipeline.




Auto-detects which feature categories apply to a video based on
content analysis, extracts every applicable feature, then
produces a verification report with confidence / quality scores
so you can confirm each extracted value is plausible.




Feature categories
──────────────────
  motion          – optical flow, freeze, burst, periodicity, rotation
  color           – per-channel means, saturation, hue distribution
  brightness      – luminance, contrast, dynamic range, histogram shape
  texture         – Laplacian sharpness, edge density, DCT entropy
  scene           – shot boundaries, shot count, scene complexity
  objects         – YOLO object detection labels, counts, confidence, positions
  audio           – (placeholder; requires ffmpeg + librosa)




Each category is auto-enabled when its detector fires above threshold.
"""




import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy.fft import dct as scipy_dct




from subject_tracker import (
    get_det_model, get_pose_model,
    ALWAYS_TRACK_CLASSES,
    CONF_THRESHOLD_PERSON, CONF_THRESHOLD_DEFAULT,
    MIN_AREA_FRACTION, MIN_FRAMES_SEEN,
    get_frame_position, get_relative_size, get_depth_hint,
    extract_pose_for_detection, build_skeleton_edges,
    detect_small_moving_objects,
)




# ══════════════════════════════════════════════════════════════════
# CATEGORY AUTO-DETECTION
# ══════════════════════════════════════════════════════════════════




def _detect_applicable_categories(cap: cv2.VideoCapture) -> dict:
    """
    Sample a handful of frames and decide which feature categories
    are meaningful for this video. Returns a dict of {category: bool}.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_indices = [int(total * p) for p in [0.1, 0.25, 0.5, 0.75, 0.9]]




    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames.append(f)




    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not frames:
        return {
            c: True for c in
            ["motion", "color", "brightness", "texture", "scene", "objects"]
        }




    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float) for f in frames]
    diffs = [float(np.abs(grays[i] - grays[i - 1]).mean()) for i in range(1, len(grays))]
    has_motion = any(d > 2.0 for d in diffs)




    sat_vals = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        sat_vals.append(float(hsv[:, :, 1].mean()))
    has_color = np.std(sat_vals) > 5 or np.mean(sat_vals) > 20




    lap_vars = [
        float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        for f in frames
    ]
    has_texture = np.mean(lap_vars) > 50




    mean_diffs = [float(np.abs(grays[i] - grays[i - 1]).mean()) for i in range(1, len(grays))]
    has_scene = any(d > 30 for d in mean_diffs) or total > fps * 5




    return {
        "motion": True,
        "color": has_color,
        "brightness": True,
        "texture": has_texture,
        "scene": has_scene,
        "objects": True,
    }




# ══════════════════════════════════════════════════════════════════
# MOTION FEATURES
# ══════════════════════════════════════════════════════════════════




def _extract_motion(cap: cv2.VideoCapture, fps: float) -> dict:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_gray = None
    flow_magnitudes = []
    frame_diffs = []
    frame_idx = 0
    per_frame = []




    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.resize(gray, (224, 224))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray_r, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
            diff = float(cv2.absdiff(prev_gray, gray_r).mean())
            flow_magnitudes.append(mag)
            frame_diffs.append(diff)
            per_frame.append({
                "frame": frame_idx,
                "mean_flow": round(mag, 4),
                "frame_diff": round(diff, 4)
            })
        prev_gray = gray_r
        frame_idx += 1




    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if len(flow_magnitudes) < 3:
        return {"error": "too few frames"}




    fa = np.array(flow_magnitudes)
    da = np.array(frame_diffs)




    fn = fa - fa.mean()
    ac = np.correlate(fn, fn, mode="full")[len(fn) - 1:]
    acn = ac / (ac[0] + 1e-8)
    periodicity = float(acn[5:20].max()) if len(acn) > 20 else 0.0




    dct_c = scipy_dct(da, norm="ortho")
    low_e = float(np.sum(dct_c[:5] ** 2))
    high_e = float(np.sum(dct_c[5:] ** 2))
    freq_ratio = high_e / (low_e + 1e-8)




    win = max(1, int(fps * 1.5))
    segs = []
    step = max(1, win // 2)
    for i in range(0, max(1, len(fa) - win + 1), step):
        chunk = fa[i:i + win]
        if len(chunk) == 0:
            continue
        segs.append({
            "t_start": round(i / fps, 2),
            "t_end": round((i + len(chunk)) / fps, 2),
            "mean_flow": round(float(chunk.mean()), 3),
            "freeze": bool(float(np.sum(chunk < 0.3) / len(chunk)) > 0.4),
            "burst": bool(float(chunk.max()) > 3.0),
        })




    half = len(fa) // 2
    return {
        "mean_flow": round(float(fa.mean()), 4),
        "max_flow": round(float(fa.max()), 4),
        "min_flow": round(float(fa.min()), 4),
        "std_flow": round(float(fa.std()), 4),
        "flow_variance": round(float(np.var(fa)), 4),
        "frame_diff_mean": round(float(da.mean()), 4),
        "frame_diff_std": round(float(da.std()), 4),
        "freeze_ratio": round(float(np.sum(fa < 0.3) / len(fa)), 4),
        "burst_ratio": round(float(np.sum(fa > 3.0) / len(fa)), 4),
        "accel_score": round(float(fa[half:].mean() - fa[:half].mean()), 4),
        "periodicity_score": round(periodicity, 4),
        "rotation_score": round(float(fa.std() / (fa.mean() + 1e-8)), 4),
        "flow_gradient": round(float(np.abs(np.diff(fa)).mean()), 4),
        "freq_ratio": round(freq_ratio, 4),
        "segments": segs,
        "per_frame": per_frame,
    }




# ══════════════════════════════════════════════════════════════════
# COLOR FEATURES
# ══════════════════════════════════════════════════════════════════




def _extract_color(cap: cv2.VideoCapture, sample_rate: int = 15) -> dict:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    r_vals, g_vals, b_vals = [], [], []
    sat_vals, hue_vals, val_vals = [], [], []
    frame_idx = 0




    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            b, g, r = cv2.split(frame)
            r_vals.append(float(r.mean()))
            g_vals.append(float(g.mean()))
            b_vals.append(float(b.mean()))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue_vals.append(float(hsv[:, :, 0].mean()))
            sat_vals.append(float(hsv[:, :, 1].mean()))
            val_vals.append(float(hsv[:, :, 2].mean()))
        frame_idx += 1




    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not r_vals:
        return {"error": "no frames sampled"}




    def stats(arr):
        a = np.array(arr)
        return {
            "mean": round(float(a.mean()), 2),
            "std": round(float(a.std()), 2),
            "min": round(float(a.min()), 2),
            "max": round(float(a.max()), 2),
        }




    hue_arr = np.array(hue_vals)
    hue_labels = ["red/orange", "yellow", "green", "cyan", "blue", "magenta"]
    hue_bucket = hue_labels[int(np.round(hue_arr.mean() / 30)) % 6]




    rb_ratio = round(float(np.mean(r_vals)) / (float(np.mean(b_vals)) + 1e-8), 3)
    kelvin = (
        2800 if rb_ratio > 1.6 else
        3500 if rb_ratio > 1.25 else
        5500 if rb_ratio > 0.9 else
        7000 if rb_ratio > 0.75 else
        9000
    )




    return {
        "red": stats(r_vals),
        "green": stats(g_vals),
        "blue": stats(b_vals),
        "hue": stats(hue_vals),
        "saturation": stats(sat_vals),
        "value": stats(val_vals),
        "dominant_hue_bucket": hue_bucket,
        "avg_rb_ratio": rb_ratio,
        "estimated_kelvin": kelvin,
        "colorfulness": round(float(np.std(r_vals + g_vals + b_vals)), 3),
    }




# ══════════════════════════════════════════════════════════════════
# BRIGHTNESS FEATURES
# ══════════════════════════════════════════════════════════════════




def _extract_brightness(cap: cv2.VideoCapture, sample_rate: int = 15) -> dict:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    lum_vals, contrast_vals, dark_pct_vals, bright_pct_vals = [], [], [], []
    frame_idx = 0




    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lum = float(gray.mean())
            lum_vals.append(lum)
            contrast_vals.append(float(gray.std()))
            dark_pct_vals.append(float((gray < 50).sum() / gray.size * 100))
            bright_pct_vals.append(float((gray > 200).sum() / gray.size * 100))
        frame_idx += 1




    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not lum_vals:
        return {"error": "no frames sampled"}




    lum_arr = np.array(lum_vals)
    avg_lum = float(lum_arr.mean())
    avg_dark = float(np.mean(dark_pct_vals))
    avg_bright = float(np.mean(bright_pct_vals))




    if avg_lum > 160 and avg_dark < 5:
        key_style = "high key"
    elif avg_lum < 80 and avg_bright < 5:
        key_style = "low key"
    elif float(np.mean(contrast_vals)) > 65:
        key_style = "high contrast (chiaroscuro)"
    elif 90 < avg_lum < 160 and float(np.mean(contrast_vals)) < 45:
        key_style = "flat / balanced"
    else:
        key_style = "mid key"




    return {
        "mean_luminance": round(avg_lum, 2),
        "std_luminance": round(float(lum_arr.std()), 2),
        "min_luminance": round(float(lum_arr.min()), 2),
        "max_luminance": round(float(lum_arr.max()), 2),
        "mean_contrast": round(float(np.mean(contrast_vals)), 2),
        "avg_dark_pct": round(avg_dark, 2),
        "avg_bright_pct": round(avg_bright, 2),
        "dynamic_range": round(float(lum_arr.max() - lum_arr.min()), 2),
        "key_style": key_style,
        "flicker_score": round(float(lum_arr.std() / (lum_arr.mean() + 1e-8)), 4),
    }




# ══════════════════════════════════════════════════════════════════
# TEXTURE / SHARPNESS FEATURES
# ══════════════════════════════════════════════════════════════════




def _extract_texture(cap: cv2.VideoCapture, sample_rate: int = 15) -> dict:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    lap_vars, edge_densities, dct_entropies = [], [], []
    frame_idx = 0




    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            edges = cv2.Canny(gray, 50, 150)
            edge_densities.append(float(edges.mean()))
            block = cv2.resize(gray, (64, 64)).astype(float)
            dct_block = scipy_dct(
                scipy_dct(block, axis=0, norm="ortho"),
                axis=1, norm="ortho"
            )
            mag = np.abs(dct_block.flatten()) + 1e-8
            mag /= mag.sum()
            entropy = float(-np.sum(mag * np.log2(mag)))
            dct_entropies.append(entropy)
        frame_idx += 1




    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not lap_vars:
        return {"error": "no frames sampled"}




    avg_lap = float(np.mean(lap_vars))
    sharpness_label = (
        "very sharp" if avg_lap > 1000 else
        "sharp" if avg_lap > 300 else
        "moderate" if avg_lap > 80 else
        "soft/blurry"
    )




    return {
        "mean_laplacian_var": round(avg_lap, 2),
        "std_laplacian_var": round(float(np.std(lap_vars)), 2),
        "mean_edge_density": round(float(np.mean(edge_densities)), 4),
        "mean_dct_entropy": round(float(np.mean(dct_entropies)), 4),
        "sharpness_label": sharpness_label,
        "texture_complexity": round(float(np.mean(dct_entropies) * np.mean(edge_densities)), 4),
    }




# ══════════════════════════════════════════════════════════════════
# SCENE / SHOT FEATURES
# ══════════════════════════════════════════════════════════════════




def _extract_scene(cap: cv2.VideoCapture, fps: float, threshold: float = 30.0) -> dict:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_hist = None
    boundaries = [0]
    frame_idx = 0
    cut_scores = []




    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None:
            diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR))
            cut_scores.append(diff)
            if diff > threshold:
                boundaries.append(frame_idx)
        prev_hist = hist
        frame_idx += 1




    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = frame_idx
    boundaries.append(total_frames)




    shots = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1] - 1
        shots.append({
            "shot_number": i + 1,
            "start_frame": s,
            "end_frame": e,
            "start_time": round(s / fps, 2),
            "end_time": round(e / fps, 2),
            "duration": round((e - s) / fps, 2),
        })




    cs = np.array(cut_scores) if cut_scores else np.array([0.0])
    return {
        "total_shots": len(shots),
        "shot_boundaries": boundaries[1:-1],
        "shots": shots,
        "avg_shot_duration": round(float(np.mean([s["duration"] for s in shots])), 2),
        "min_shot_duration": round(float(min(s["duration"] for s in shots)), 2),
        "max_shot_duration": round(float(max(s["duration"] for s in shots)), 2),
        "avg_cut_score": round(float(cs.mean()), 3),
        "max_cut_score": round(float(cs.max()), 3),
        "scene_complexity": round(float(len(shots) / max(total_frames / fps, 1)), 4),
    }




# ══════════════════════════════════════════════════════════════════
# OBJECT FEATURES
# ══════════════════════════════════════════════════════════════════

# Relaxed temporal gate for the feature extractor — looser than subject_tracker
# so briefly-appearing objects (yo-yo, thrown ball, etc.) are not dropped.
_OBJ_MIN_FRAMES = {
    "person":       1,
    "moving_object": 1,
    "default":      1,
}


def _extract_objects(
    cap: cv2.VideoCapture,
    fps: float,
    sample_rate: int = 15,
    conf_threshold: float = 0.25,
    max_frames: int = 300
) -> dict:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    det_model  = get_det_model()
    pose_model = get_pose_model()




    frame_idx      = 0
    sampled_frames = 0
    all_detections = []
    frames_with_objects = []
    label_frequency    = Counter()
    label_confidences  = defaultdict(list)
    label_frames       = defaultdict(list)
    size_counts        = defaultdict(Counter)
    position_counts    = defaultdict(Counter)
    depth_counts       = defaultdict(Counter)
    prev_gray_obj      = None   # for background-subtraction small-object pass




    while True:
        ret, frame = cap.read()
        if not ret:
            break




        if frame_idx % sample_rate == 0:
            fh, fw     = frame.shape[:2]
            frame_area = fw * fh

            # ── YOLO detection pass ───────────────────────────────
            # Run at person threshold so all candidates come through;
            # per-class gate is applied manually below (Fix 1).
            det_results = det_model(frame, verbose=False, conf=CONF_THRESHOLD_PERSON)[0]
            try:
                pose_results = pose_model(frame, verbose=False, conf=CONF_THRESHOLD_PERSON)[0]
            except Exception:
                pose_results = None

            detections             = []
            seen_labels_this_frame = set()

            for box in det_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf  = float(box.conf[0])
                label = det_model.names[int(box.cls[0])]

                # Fix 1 — dual confidence gate
                req_conf = (CONF_THRESHOLD_PERSON
                            if label in ALWAYS_TRACK_CLASSES
                            else CONF_THRESHOLD_DEFAULT)
                if conf < req_conf:
                    continue

                w  = x2 - x1
                h  = y2 - y1
                cx = x1 + w // 2
                cy = y1 + h // 2

                # Fix 2 — minimum bounding-box area gate
                area_frac = (w * h) / (frame_area + 1e-8)
                min_frac  = MIN_AREA_FRACTION.get(label, MIN_AREA_FRACTION["default"])
                if area_frac < min_frac:
                    continue

                bbox = {"x": x1, "y": y1, "w": w, "h": h,
                        "x2": x2, "y2": y2, "cx": cx, "cy": cy}

                det = {
                    "label":             label,
                    "confidence":        round(conf, 3),
                    "bbox":              bbox,
                    "position_in_frame": get_frame_position(cx, cy, fw, fh),
                    "relative_size":     get_relative_size(w * h, frame_area),
                    "depth_hint":        get_depth_hint(w * h, frame_area, cy, fh),
                    "pose":              None,
                }

                if label == "person" and pose_results is not None:
                    kpts = extract_pose_for_detection(pose_results, bbox)
                    if kpts:
                        det["pose"] = {
                            "keypoints":      kpts,
                            "skeleton_edges": build_skeleton_edges(kpts),
                            "visible_joints": sum(1 for kp in kpts if kp["visible"]),
                            "total_joints":   len(kpts),
                        }

                detections.append(det)
                if label not in seen_labels_this_frame:
                    seen_labels_this_frame.add(label)

            # ── Background subtraction pass ───────────────────────
            # Catches yo-yo / small fast objects that YOLO misses.
            # Only runs when we have a previous sampled frame to diff against.
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray_obj is not None:
                small_dets = detect_small_moving_objects(
                    prev_gray_obj, curr_gray, fw, fh,
                    label="moving_object"
                )
                for sd in small_dets:
                    sx, sy = sd["bbox"]["cx"], sd["bbox"]["cy"]
                    # Skip if YOLO already detected something in roughly the same spot
                    already_covered = any(
                        abs(sx - d["bbox"]["cx"]) < 30 and
                        abs(sy - d["bbox"]["cy"]) < 30
                        for d in detections
                    )
                    if not already_covered:
                        detections.append(sd)
                        if "moving_object" not in seen_labels_this_frame:
                            seen_labels_this_frame.add("moving_object")

            prev_gray_obj = curr_gray




            if detections:
                ts = round(frame_idx / fps, 2)
                frames_with_objects.append({
                    "frame": frame_idx,
                    "timestamp_sec": ts,
                    "labels_present": sorted(list(set(d["label"] for d in detections))),
                    "object_count": len(detections),
                    "detections": [
                        {
                            "label": d["label"],
                            "confidence": d["confidence"],
                            "bbox": d["bbox"],
                            "position_in_frame": d["position_in_frame"],
                            "relative_size": d["relative_size"],
                            "depth_hint": d["depth_hint"],
                        }
                        for d in detections
                    ]
                })




                for d in detections:
                    label = d["label"]
                    label_frequency[label] += 1
                    label_confidences[label].append(float(d["confidence"]))
                    label_frames[label].append(frame_idx)
                    size_counts[label][d["relative_size"]] += 1
                    position_counts[label][d["position_in_frame"]] += 1
                    depth_counts[label][d["depth_hint"]] += 1
                    all_detections.append(d)




            sampled_frames += 1
            if sampled_frames >= max_frames:
                break




        frame_idx += 1




    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Fix 3 — temporal gate: use _OBJ_MIN_FRAMES (relaxed) instead of
    # MIN_FRAMES_SEEN so briefly-appearing objects are not dropped.
    filtered_label_frequency = Counter({
        label: count
        for label, count in label_frequency.items()
        if len(set(label_frames[label])) >= _OBJ_MIN_FRAMES.get(label, _OBJ_MIN_FRAMES["default"])
    })

    unique_labels    = sorted(filtered_label_frequency.keys())
    dominant_objects = [label for label, _ in filtered_label_frequency.most_common(10)]
    total_dets_filt  = sum(filtered_label_frequency.values())




    object_summary = {}
    for label, count in filtered_label_frequency.most_common():
        frames_seen = sorted(set(label_frames[label]))
        object_summary[label] = {
            "detections": int(count),
            "frames_seen": len(frames_seen),
            "first_frame": frames_seen[0] if frames_seen else None,
            "last_frame": frames_seen[-1] if frames_seen else None,
            "first_timestamp_sec": round(frames_seen[0] / fps, 2) if frames_seen else None,
            "last_timestamp_sec": round(frames_seen[-1] / fps, 2) if frames_seen else None,
            "avg_confidence": round(float(np.mean(label_confidences[label])), 3),
            "max_confidence": round(float(np.max(label_confidences[label])), 3),
            "dominant_size": size_counts[label].most_common(1)[0][0] if size_counts[label] else None,
            "dominant_position": position_counts[label].most_common(1)[0][0] if position_counts[label] else None,
            "dominant_depth": depth_counts[label].most_common(1)[0][0] if depth_counts[label] else None,
        }




    return {
        "sample_rate": sample_rate,
        "sampled_frames": sampled_frames,
        "total_detections": total_dets_filt,
        "unique_labels": unique_labels,
        "dominant_objects": dominant_objects,
        "label_frequency": dict(filtered_label_frequency.most_common()),
        "object_summary": object_summary,
        "frames_with_objects": frames_with_objects,
        "avg_detections_per_sampled_frame": round(
            total_dets_filt / max(sampled_frames, 1), 3
        ),
    }




# ══════════════════════════════════════════════════════════════════
# VERIFICATION ENGINE
# ══════════════════════════════════════════════════════════════════




def _verify_features(features: dict, categories: dict, fps: float, total_frames: int) -> dict:
    """
    For each extracted feature group, produce a verification dict:
    - status: "pass" | "warn" | "fail"
    - checks: list of individual check results
    - confidence: 0.0–1.0
    """
    duration = total_frames / fps
    results = {}




    if categories.get("motion") and "motion" in features:
        m = features["motion"]
        checks = []
        passes = 0




        mf = m.get("mean_flow", -1)
        ok = 0 <= mf <= 50
        checks.append({
            "feature": "mean_flow",
            "value": mf,
            "expected": "0–50",
            "pass": ok,
            "note": "optical flow magnitude, typical videos 0.2–5"
        })
        if ok:
            passes += 1




        fr = m.get("freeze_ratio", 0)
        br = m.get("burst_ratio", 0)
        ok = 0 <= fr + br <= 1.0
        checks.append({
            "feature": "freeze+burst_ratio",
            "value": round(fr + br, 3),
            "expected": "≤1.0",
            "pass": ok,
            "note": "fractions of frames, must not exceed 1"
        })
        if ok:
            passes += 1




        ps = m.get("periodicity_score", -1)
        ok = 0 <= ps <= 1
        checks.append({
            "feature": "periodicity_score",
            "value": ps,
            "expected": "0–1",
            "pass": ok,
            "note": "autocorrelation peak, 1=perfectly periodic"
        })
        if ok:
            passes += 1




        seg_count = len(m.get("segments", []))
        expected_min = max(1, int(duration / 3))
        ok = seg_count >= expected_min
        checks.append({
            "feature": "segment_count",
            "value": seg_count,
            "expected": f"≥{expected_min}",
            "pass": ok,
            "note": "sliding window segments for a clip this length"
        })
        if ok:
            passes += 1




        pf_count = len(m.get("per_frame", []))
        ok = abs(pf_count - (total_frames - 1)) <= 2
        checks.append({
            "feature": "per_frame_count",
            "value": pf_count,
            "expected": f"~{total_frames - 1}",
            "pass": ok,
            "note": "one flow measurement per consecutive frame pair"
        })
        if ok:
            passes += 1




        conf = passes / len(checks)
        results["motion"] = {
            "status": "pass" if conf >= 0.8 else "warn" if conf >= 0.5 else "fail",
            "confidence": round(conf, 2),
            "checks": checks,
        }




    if categories.get("color") and "color" in features:
        c = features["color"]
        checks = []
        passes = 0




        for ch in ["red", "green", "blue"]:
            mean_val = c.get(ch, {}).get("mean", -1)
            ok = 0 <= mean_val <= 255
            checks.append({
                "feature": f"{ch}_mean",
                "value": mean_val,
                "expected": "0–255",
                "pass": ok,
                "note": "channel mean must be valid pixel value"
            })
            if ok:
                passes += 1




        sat_mean = c.get("saturation", {}).get("mean", -1)
        ok = 0 <= sat_mean <= 255
        checks.append({
            "feature": "saturation_mean",
            "value": sat_mean,
            "expected": "0–255",
            "pass": ok,
            "note": "HSV saturation range"
        })
        if ok:
            passes += 1




        kel = c.get("estimated_kelvin", 0)
        ok = 2500 <= kel <= 10000
        checks.append({
            "feature": "estimated_kelvin",
            "value": kel,
            "expected": "2500–10000",
            "pass": ok,
            "note": "plausible colour temperature range for video"
        })
        if ok:
            passes += 1




        conf = passes / len(checks)
        results["color"] = {
            "status": "pass" if conf >= 0.8 else "warn" if conf >= 0.5 else "fail",
            "confidence": round(conf, 2),
            "checks": checks,
        }




    if categories.get("brightness") and "brightness" in features:
        b = features["brightness"]
        checks = []
        passes = 0




        lum = b.get("mean_luminance", -1)
        ok = 0 <= lum <= 255
        checks.append({
            "feature": "mean_luminance",
            "value": lum,
            "expected": "0–255",
            "pass": ok,
            "note": "average pixel brightness"
        })
        if ok:
            passes += 1




        dr = b.get("dynamic_range", -1)
        ok = 0 <= dr <= 255
        checks.append({
            "feature": "dynamic_range",
            "value": dr,
            "expected": "0–255",
            "pass": ok,
            "note": "max−min luminance across sampled frames"
        })
        if ok:
            passes += 1




        dark_pct = b.get("avg_dark_pct", -1)
        bright_pct = b.get("avg_bright_pct", -1)
        ok = 0 <= dark_pct + bright_pct <= 100
        checks.append({
            "feature": "dark+bright_pct",
            "value": round(dark_pct + bright_pct, 1),
            "expected": "0–100",
            "pass": ok,
            "note": "fractions of very dark + very bright pixels"
        })
        if ok:
            passes += 1




        flicker = b.get("flicker_score", -1)
        ok = 0 <= flicker <= 5
        checks.append({
            "feature": "flicker_score",
            "value": flicker,
            "expected": "0–5",
            "pass": ok,
            "note": "luminance std/mean; >1 suggests strobing"
        })
        if ok:
            passes += 1




        conf = passes / len(checks)
        results["brightness"] = {
            "status": "pass" if conf >= 0.8 else "warn" if conf >= 0.5 else "fail",
            "confidence": round(conf, 2),
            "checks": checks,
        }




    if categories.get("texture") and "texture" in features:
        t = features["texture"]
        checks = []
        passes = 0




        lap = t.get("mean_laplacian_var", -1)
        ok = lap >= 0
        checks.append({
            "feature": "mean_laplacian_var",
            "value": lap,
            "expected": "≥0",
            "pass": ok,
            "note": "≈0=blurry, 100+=moderate, 1000+=very sharp"
        })
        if ok:
            passes += 1




        ed = t.get("mean_edge_density", -1)
        ok = 0 <= ed <= 255
        checks.append({
            "feature": "mean_edge_density",
            "value": ed,
            "expected": "0–255",
            "pass": ok,
            "note": "average Canny edge pixel intensity"
        })
        if ok:
            passes += 1




        ent = t.get("mean_dct_entropy", -1)
        ok = 0 <= ent <= 20
        checks.append({
            "feature": "mean_dct_entropy",
            "value": ent,
            "expected": "0–20",
            "pass": ok,
            "note": "DCT coefficient entropy; higher = more complex texture"
        })
        if ok:
            passes += 1




        conf = passes / len(checks)
        results["texture"] = {
            "status": "pass" if conf >= 0.8 else "warn" if conf >= 0.5 else "fail",
            "confidence": round(conf, 2),
            "checks": checks,
        }




    if categories.get("scene") and "scene" in features:
        s = features["scene"]
        checks = []
        passes = 0




        shots = s.get("total_shots", 0)
        ok = shots >= 1
        checks.append({
            "feature": "total_shots",
            "value": shots,
            "expected": "≥1",
            "pass": ok,
            "note": "at least one shot always exists"
        })
        if ok:
            passes += 1




        avg_dur = s.get("avg_shot_duration", -1)
        ok = 0 < avg_dur <= duration
        checks.append({
            "feature": "avg_shot_duration",
            "value": avg_dur,
            "expected": f"0–{duration}s",
            "pass": ok,
            "note": "average shot length must fit inside clip"
        })
        if ok:
            passes += 1




        min_dur = s.get("min_shot_duration", -1)
        ok = min_dur >= 0
        checks.append({
            "feature": "min_shot_duration",
            "value": min_dur,
            "expected": "≥0",
            "pass": ok,
            "note": "shortest shot non-negative"
        })
        if ok:
            passes += 1




        conf = passes / len(checks)
        results["scene"] = {
            "status": "pass" if conf >= 0.8 else "warn" if conf >= 0.5 else "fail",
            "confidence": round(conf, 2),
            "checks": checks,
        }




    if categories.get("objects") and "objects" in features:
        o = features["objects"]
        checks = []
        passes = 0




        sampled_frames = o.get("sampled_frames", 0)
        ok = sampled_frames >= 1
        checks.append({
            "feature": "sampled_frames",
            "value": sampled_frames,
            "expected": "≥1",
            "pass": ok,
            "note": "object detector should analyze at least one sampled frame"
        })
        if ok:
            passes += 1




        total_detections = o.get("total_detections", -1)
        ok = total_detections >= 0
        checks.append({
            "feature": "total_detections",
            "value": total_detections,
            "expected": "≥0",
            "pass": ok,
            "note": "object count cannot be negative"
        })
        if ok:
            passes += 1




        avg_det = o.get("avg_detections_per_sampled_frame", -1)
        ok = avg_det >= 0
        checks.append({
            "feature": "avg_detections_per_sampled_frame",
            "value": avg_det,
            "expected": "≥0",
            "pass": ok,
            "note": "average detections per sampled frame must be non-negative"
        })
        if ok:
            passes += 1




        label_freq = o.get("label_frequency", {})
        sum_freq = sum(label_freq.values()) if isinstance(label_freq, dict) else -1
        ok = sum_freq == total_detections
        checks.append({
            "feature": "label_frequency_sum",
            "value": sum_freq,
            "expected": f"{total_detections}",
            "pass": ok,
            "note": "sum of per-label detection counts should equal total detections"
        })
        if ok:
            passes += 1




        conf = passes / len(checks)
        results["objects"] = {
            "status": "pass" if conf >= 0.8 else "warn" if conf >= 0.5 else "fail",
            "confidence": round(conf, 2),
            "checks": checks,
        }




    return results




# ══════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════




def extract_all_features(video_path: str, progress_cb=None) -> dict:
    """
    Main entry point. Pass a video path, get back:
      {
        "file_info":    {...},
        "categories":   {"motion": true, "color": true, ...},
        "features":     {"motion": {...}, "color": {...}, ...},
        "verification": {"motion": {"status":"pass","confidence":0.9,"checks":[...]}, ...},
        "summary":      {"total_features": N, "passed": N, "warned": N, "failed": N}
      }
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))




    file_info = {
        "file": Path(video_path).name,
        "fps": round(fps, 2),
        "total_frames": total,
        "duration_sec": round(total / fps, 2) if fps else 0,
        "resolution": f"{w}×{h}",
        "aspect_ratio": round(w / h, 3) if h else 0,
    }




    if progress_cb:
        progress_cb(5)




    categories = _detect_applicable_categories(cap)




    if progress_cb:
        progress_cb(10)




    features = {}
    steps = [c for c, en in categories.items() if en]
    n_steps = len(steps)




    extractors = {
        "motion": lambda: _extract_motion(cap, fps),
        "color": lambda: _extract_color(cap),
        "brightness": lambda: _extract_brightness(cap),
        "texture": lambda: _extract_texture(cap),
        "scene": lambda: _extract_scene(cap, fps),
        "objects": lambda: _extract_objects(cap, fps),
    }




    for i, cat in enumerate(steps):
        if cat in extractors:
            features[cat] = extractors[cat]()
        if progress_cb and n_steps > 0:
            progress_cb(10 + int((i + 1) / n_steps * 75))




    cap.release()




    verification = _verify_features(features, categories, fps, total)
    if progress_cb:
        progress_cb(95)




    statuses = [v["status"] for v in verification.values()]
    n_checks = sum(len(v["checks"]) for v in verification.values())
    n_passed = sum(1 for v in verification.values() for c in v["checks"] if c["pass"])
    summary = {
        "total_feature_groups": len(features),
        "total_checks": n_checks,
        "passed_checks": n_passed,
        "warned_groups": statuses.count("warn"),
        "failed_groups": statuses.count("fail"),
        "overall_confidence": round(n_passed / max(n_checks, 1), 3),
    }




    if progress_cb:
        progress_cb(100)




    return {
        "file_info": file_info,
        "categories": categories,
        "features": features,
        "verification": verification,
        "summary": summary,
    }