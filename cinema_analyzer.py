import cv2
import numpy as np
from pathlib import Path

# ── Color temperature ──────────────────────────────────────────────
def estimate_color_temp(frame_bgr):
    """
    Warm = high R relative to B → tungsten/golden hour
    Cool = high B relative to R → daylight/overcast/LED
    Returns: kelvin estimate + label
    """
    b = float(frame_bgr[:,:,0].mean())
    g = float(frame_bgr[:,:,1].mean())
    r = float(frame_bgr[:,:,2].mean())
    ratio = r / (b + 1e-8)
    if   ratio > 1.6:  kelvin, label = 2800, "very warm (tungsten/candlelight)"
    elif ratio > 1.25: kelvin, label = 3500, "warm (golden hour / indoor)"
    elif ratio > 0.9:  kelvin, label = 5500, "neutral (daylight)"
    elif ratio > 0.75: kelvin, label = 7000, "cool (overcast / shade)"
    else:              kelvin, label = 9000, "very cool (blue hour / LED)"
    return {"kelvin": kelvin, "label": label, "r_mean": round(r,1),
            "g_mean": round(g,1), "b_mean": round(b,1), "rb_ratio": round(ratio,3)}

# ── Natural vs artificial ──────────────────────────────────────────
def classify_light_source(frame_bgr, temp_data):
    """
    Heuristics:
    - Artificial: very warm OR very cool + high contrast + uneven falloff
    - Natural: neutral-to-warm + soft shadows + gradient brightness
    """
    gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    contrast  = float(gray.std())
    mean_lum  = float(gray.mean())
    # Frequency of near-saturation pixels (blown highlights = artificial point source)
    highlight_pct = float((gray > 240).sum() / gray.size)

    kelvin = temp_data["kelvin"]
    if kelvin <= 3500 and contrast > 55:
        source, confidence = "artificial (tungsten/warm LED)", 0.82
    elif kelvin >= 7000 and highlight_pct < 0.01:
        source, confidence = "artificial (cool LED / fluorescent)", 0.78
    elif 4500 <= kelvin <= 6500 and contrast < 50:
        source, confidence = "natural (overcast / diffuse daylight)", 0.80
    elif 3500 < kelvin < 5500 and 0.01 < highlight_pct < 0.08:
        source, confidence = "natural (direct sunlight / golden hour)", 0.85
    else:
        source, confidence = "mixed / ambiguous", 0.55

    return {"source": source, "confidence": round(confidence, 2),
            "contrast": round(contrast, 2), "mean_luminance": round(mean_lum, 2),
            "highlight_pct": round(highlight_pct * 100, 2)}

# ── Light direction ────────────────────────────────────────────────
def estimate_light_direction(frame_bgr):
    """
    Divide frame into zones and find brightest region.
    Infer light source position from brightest zone.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    h, w = gray.shape
    zones = {
        "top":    gray[:h//3, :].mean(),
        "bottom": gray[2*h//3:, :].mean(),
        "left":   gray[:, :w//3].mean(),
        "right":  gray[:, 2*w//3:].mean(),
        "center": gray[h//4:3*h//4, w//4:3*w//4].mean(),
    }
    brightest = max(zones, key=zones.get)
    darkest   = min(zones, key=zones.get)
    ratio     = zones[brightest] / (zones[darkest] + 1e-8)

    direction_map = {
        "top":    "top-lit (overhead / ceiling light / noon sun)",
        "bottom": "bottom-lit (floor reflection / uplighting)",
        "left":   "side-lit from left",
        "right":  "side-lit from right",
        "center": "front-lit (flat lighting / ring light)",
    }

    # Backlight: edges brighter than center
    edge_mean   = np.mean([zones["top"], zones["bottom"], zones["left"], zones["right"]])
    center_mean = zones["center"]
    if edge_mean > center_mean * 1.3:
        direction = "backlit (silhouette / rim light)"
        brightest = "edges"
    else:
        direction = direction_map.get(brightest, "unknown")

    return {
        "direction":    direction,
        "brightest_zone": brightest,
        "contrast_ratio": round(ratio, 2),
        "zone_means":   {k: round(v, 1) for k, v in zones.items()},
    }

# ── Floor / surface reflection ─────────────────────────────────────
def detect_floor_reflection(frame_bgr):
    """
    Reflections in bottom third: high brightness + color similarity to upper zones.
    """
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    bottom = gray[2*h//3:, :]
    upper  = gray[:h//3, :]

    bottom_bright = float(bottom.mean())
    upper_bright  = float(upper.mean())

    # Color correlation between bottom and upper (reflection = similar hue)
    b_bottom = frame_bgr[2*h//3:, :, 0].mean()
    r_bottom = frame_bgr[2*h//3:, :, 2].mean()
    b_upper  = frame_bgr[:h//3, :, 0].mean()
    r_upper  = frame_bgr[:h//3, :, 2].mean()
    color_sim = 1 - abs(b_bottom/( b_upper+1e-8) - 1) - abs(r_bottom/(r_upper+1e-8) - 1)
    color_sim = max(0, min(1, color_sim))

    # High local variance in bottom = specular highlight / glare
    bottom_var = float(bottom.var())

    reflection_score = (
        0.4 * min(bottom_bright / (upper_bright + 1e-8), 1.5) / 1.5 +
        0.4 * color_sim +
        0.2 * min(bottom_var / 1000, 1.0)
    )

    detected = reflection_score > 0.45
    if detected:
        if bottom_var > 800:
            rtype = "specular (glare / wet floor)"
        elif color_sim > 0.7:
            rtype = "diffuse bounce (soft floor reflection)"
        else:
            rtype = "partial reflection"
    else:
        rtype = "none detected"

    return {
        "reflection_detected": detected,
        "reflection_type":     rtype,
        "reflection_score":    round(reflection_score, 3),
        "bottom_brightness":   round(bottom_bright, 1),
        "color_similarity":    round(color_sim, 3),
    }

# ── High key vs low key ────────────────────────────────────────────
def classify_key(frame_bgr):
    gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum  = float(gray.mean())
    dark_pct  = float((gray < 50).sum()  / gray.size * 100)
    light_pct = float((gray > 200).sum() / gray.size * 100)
    contrast  = float(gray.std())

    if mean_lum > 160 and dark_pct < 5:
        style = "high key (bright, minimal shadows)"
    elif mean_lum < 80 and light_pct < 5:
        style = "low key (dark, dramatic shadows)"
    elif contrast > 65:
        style = "high contrast (chiaroscuro)"
    elif 90 < mean_lum < 160 and contrast < 45:
        style = "flat / balanced"
    else:
        style = "mid key (standard exposure)"

    return {
        "style":       style,
        "mean_lum":    round(mean_lum, 1),
        "dark_pct":    round(dark_pct, 1),
        "light_pct":   round(light_pct, 1),
        "contrast_std": round(contrast, 1),
    }

# ── Camera angle ───────────────────────────────────────────────────
def estimate_camera_angle(frame_bgr, flow_data=None):
    """
    Horizon line position → tilt angle.
    Edge density distribution → shot type hint.
    """
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w    = gray.shape
    edges   = cv2.Canny(gray, 50, 150)

    # Horizon estimation: find row with most horizontal edges
    horiz_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    horiz_edges     = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horiz_kernel)
    row_sums        = horiz_edges.sum(axis=1)
    horizon_row     = int(row_sums.argmax())
    horizon_pct     = horizon_row / h

    if   horizon_pct < 0.3: angle = "high angle (camera looking down)"
    elif horizon_pct > 0.7: angle = "low angle (camera looking up)"
    elif 0.45 < horizon_pct < 0.55: angle = "eye level"
    else: angle = "slight tilt / three-quarter angle"

    # Dutch tilt: detect diagonal dominant edges
    diag_mask = np.zeros_like(edges)
    cv2.line(diag_mask, (0,0), (w,h), 255, 20)
    cv2.line(diag_mask, (w,0), (0,h), 255, 20)
    diag_score = float((edges & diag_mask).sum()) / (edges.sum() + 1e-8)
    dutch_tilt = diag_score > 0.35

    # Shot type: edge density in center vs frame
    center_edges = edges[h//4:3*h//4, w//4:3*w//4]
    edge_density_center = float(center_edges.mean())
    edge_density_full   = float(edges.mean())
    center_ratio        = edge_density_center / (edge_density_full + 1e-8)

    if   center_ratio > 1.6: shot_type = "close-up (subject fills frame)"
    elif center_ratio > 1.2: shot_type = "medium shot"
    elif center_ratio < 0.8: shot_type = "wide shot (subject small in frame)"
    else:                    shot_type = "medium-wide shot"

    return {
        "angle":         ("dutch tilt" if dutch_tilt else angle),
        "shot_type":     shot_type,
        "horizon_pct":   round(horizon_pct, 3),
        "dutch_tilt":    dutch_tilt,
        "diag_score":    round(diag_score, 3),
    }

# ── Camera movement ────────────────────────────────────────────────
def classify_camera_movement(flow_magnitudes, flow_directions=None):
    if not flow_magnitudes or len(flow_magnitudes) < 3:
        return {"movement": "unknown", "shake_score": 0.0}

    arr   = np.array(flow_magnitudes)
    mean  = arr.mean()
    std   = arr.std()
    shake = float(std / (mean + 1e-8))

    if   mean < 0.3:                    movement = "static (locked-off)"
    elif shake > 1.2:                   movement = "handheld (shaky)"
    elif mean > 2.5 and shake < 0.6:    movement = "fast pan / tracking"
    elif mean > 1.0 and shake < 0.5:    movement = "slow pan / dolly"
    elif std < 0.3 and mean > 0.5:      movement = "smooth zoom"
    else:                               movement = "mixed / organic movement"

    return {"movement": movement, "shake_score": round(shake, 3),
            "mean_flow": round(float(mean), 3)}

# ── Full cinema analysis for one frame ────────────────────────────
def analyze_frame(frame_bgr):
    temp    = estimate_color_temp(frame_bgr)
    source  = classify_light_source(frame_bgr, temp)
    dirn    = estimate_light_direction(frame_bgr)
    reflect = detect_floor_reflection(frame_bgr)
    key     = classify_key(frame_bgr)
    cam     = estimate_camera_angle(frame_bgr)
    return {
        "color_temperature": temp,
        "light_source":      source,
        "light_direction":   dirn,
        "floor_reflection":  reflect,
        "key_style":         key,
        "camera_angle":      cam,
    }

# ── Full video cinema analysis ─────────────────────────────────────
def analyze_video_cinema(video_path: str, sample_rate: int = 15):
    """
    Sample every `sample_rate` frames.
    Returns per-clip summary + per-segment breakdown.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_idx     = 0
    sampled       = []
    flow_mags     = []
    prev_gray     = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            flow_mags.append(float(mag.mean()))

        if frame_idx % sample_rate == 0:
            analysis = analyze_frame(frame)
            analysis["timestamp_sec"] = round(frame_idx / fps, 2)
            analysis["frame_idx"]     = frame_idx
            sampled.append(analysis)

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not sampled:
        return None

    # ── Aggregate per-clip summary ──
    def most_common(lst):
        from collections import Counter
        return Counter(lst).most_common(1)[0][0] if lst else "unknown"

    def safe_mean(lst):
        return round(float(np.mean(lst)), 3) if lst else 0.0

    summary = {
        "dominant_light_source":    most_common([s["light_source"]["source"]    for s in sampled]),
        "dominant_light_direction": most_common([s["light_direction"]["direction"] for s in sampled]),
        "dominant_key_style":       most_common([s["key_style"]["style"]         for s in sampled]),
        "dominant_camera_angle":    most_common([s["camera_angle"]["angle"]      for s in sampled]),
        "dominant_shot_type":       most_common([s["camera_angle"]["shot_type"]  for s in sampled]),
        "floor_reflection_detected": any(s["floor_reflection"]["reflection_detected"] for s in sampled),
        "floor_reflection_type":    most_common([s["floor_reflection"]["reflection_type"] for s in sampled]),
        "avg_color_temp_kelvin":    safe_mean([s["color_temperature"]["kelvin"]  for s in sampled]),
        "avg_luminance":            safe_mean([s["key_style"]["mean_lum"]        for s in sampled]),
        "avg_contrast":             safe_mean([s["key_style"]["contrast_std"]    for s in sampled]),
        "camera_movement":          classify_camera_movement(flow_mags),
    }

    # ── Per-segment breakdown (group every 5 samples) ──
    seg_size = 5
    segments = []
    for i in range(0, len(sampled), seg_size):
        chunk = sampled[i:i + seg_size]
        t_start = chunk[0]["timestamp_sec"]
        t_end   = chunk[-1]["timestamp_sec"]
        seg_flows = flow_mags[
            int(t_start * fps):int(t_end * fps)
        ] if flow_mags else []

        segments.append({
            "t_start":        t_start,
            "t_end":          t_end,
            "light_source":   most_common([c["light_source"]["source"]      for c in chunk]),
            "light_direction":most_common([c["light_direction"]["direction"] for c in chunk]),
            "key_style":      most_common([c["key_style"]["style"]           for c in chunk]),
            "camera_angle":   most_common([c["camera_angle"]["angle"]        for c in chunk]),
            "shot_type":      most_common([c["camera_angle"]["shot_type"]    for c in chunk]),
            "reflection":     any(c["floor_reflection"]["reflection_detected"] for c in chunk),
            "avg_kelvin":     safe_mean([c["color_temperature"]["kelvin"]    for c in chunk]),
            "avg_luminance":  safe_mean([c["key_style"]["mean_lum"]          for c in chunk]),
            "camera_movement": classify_camera_movement(seg_flows)["movement"],
        })

    return {"summary": summary, "segments": segments, "frame_samples": sampled}