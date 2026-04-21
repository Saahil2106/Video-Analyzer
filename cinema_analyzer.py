import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════

def most_common(lst):
    return Counter(lst).most_common(1)[0][0] if lst else "unknown"

def safe_mean(lst):
    return round(float(np.mean(lst)), 3) if lst else 0.0

# ══════════════════════════════════════════════════════════════════
# COLOR TEMPERATURE
# ══════════════════════════════════════════════════════════════════

def estimate_color_temp(frame_bgr):
    b = float(frame_bgr[:,:,0].mean())
    g = float(frame_bgr[:,:,1].mean())
    r = float(frame_bgr[:,:,2].mean())
    ratio = r / (b + 1e-8)
    if   ratio > 1.6:  kelvin, label = 2800, "very warm (tungsten/candlelight)"
    elif ratio > 1.25: kelvin, label = 3500, "warm (golden hour / indoor)"
    elif ratio > 0.9:  kelvin, label = 5500, "neutral (daylight)"
    elif ratio > 0.75: kelvin, label = 7000, "cool (overcast / shade)"
    else:              kelvin, label = 9000, "very cool (blue hour / LED)"
    return {
        "kelvin":   kelvin,
        "label":    label,
        "r_mean":   round(r, 1),
        "g_mean":   round(g, 1),
        "b_mean":   round(b, 1),
        "rb_ratio": round(ratio, 3),
    }

# ══════════════════════════════════════════════════════════════════
# NATURAL VS ARTIFICIAL
# ══════════════════════════════════════════════════════════════════

def classify_light_source(frame_bgr, temp_data):
    gray          = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    contrast      = float(gray.std())
    mean_lum      = float(gray.mean())
    highlight_pct = float((gray > 240).sum() / gray.size)
    kelvin        = temp_data["kelvin"]

    if   kelvin <= 3500 and contrast > 55:
        source, confidence = "artificial (tungsten/warm LED)", 0.82
    elif kelvin >= 7000 and highlight_pct < 0.01:
        source, confidence = "artificial (cool LED / fluorescent)", 0.78
    elif 4500 <= kelvin <= 6500 and contrast < 50:
        source, confidence = "natural (overcast / diffuse daylight)", 0.80
    elif 3500 < kelvin < 5500 and 0.01 < highlight_pct < 0.08:
        source, confidence = "natural (direct sunlight / golden hour)", 0.85
    else:
        source, confidence = "mixed / ambiguous", 0.55

    return {
        "source":         source,
        "confidence":     round(confidence, 2),
        "contrast":       round(contrast, 2),
        "mean_luminance": round(mean_lum, 2),
        "highlight_pct":  round(highlight_pct * 100, 2),
    }

# ══════════════════════════════════════════════════════════════════
# FRAME-LEVEL LIGHT DIRECTION
# ══════════════════════════════════════════════════════════════════

def estimate_light_direction(frame_bgr):
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

    edge_mean   = np.mean([zones["top"], zones["bottom"],
                           zones["left"], zones["right"]])
    center_mean = zones["center"]
    if edge_mean > center_mean * 1.3:
        direction = "backlit (silhouette / rim light)"
        brightest = "edges"
    else:
        direction_map = {
            "top":    "top-lit (overhead / ceiling light / noon sun)",
            "bottom": "bottom-lit (floor reflection / uplighting)",
            "left":   "side-lit from left",
            "right":  "side-lit from right",
            "center": "front-lit (flat lighting / ring light)",
        }
        direction = direction_map.get(brightest, "unknown")

    return {
        "direction":      direction,
        "brightest_zone": brightest,
        "contrast_ratio": round(ratio, 2),
        "zone_means":     {k: round(v, 1) for k, v in zones.items()},
    }

# ══════════════════════════════════════════════════════════════════
# FLOOR / SURFACE REFLECTION
# ══════════════════════════════════════════════════════════════════

def detect_floor_reflection(frame_bgr):
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    bottom = gray[2*h//3:, :]
    upper  = gray[:h//3, :]

    bottom_bright = float(bottom.mean())
    upper_bright  = float(upper.mean())
    b_bottom = frame_bgr[2*h//3:, :, 0].mean()
    r_bottom = frame_bgr[2*h//3:, :, 2].mean()
    b_upper  = frame_bgr[:h//3, :, 0].mean()
    r_upper  = frame_bgr[:h//3, :, 2].mean()
    color_sim = 1 - abs(b_bottom/(b_upper+1e-8)-1) - abs(r_bottom/(r_upper+1e-8)-1)
    color_sim = max(0, min(1, color_sim))
    bottom_var = float(bottom.var())

    reflection_score = (
        0.4 * min(bottom_bright / (upper_bright + 1e-8), 1.5) / 1.5 +
        0.4 * color_sim +
        0.2 * min(bottom_var / 1000, 1.0)
    )
    detected = reflection_score > 0.45
    if detected:
        rtype = ("specular (glare / wet floor)" if bottom_var > 800
                 else "diffuse bounce (soft floor reflection)"
                 if color_sim > 0.7 else "partial reflection")
    else:
        rtype = "none detected"

    return {
        "reflection_detected": detected,
        "reflection_type":     rtype,
        "reflection_score":    round(reflection_score, 3),
        "bottom_brightness":   round(bottom_bright, 1),
        "color_similarity":    round(color_sim, 3),
    }

# ══════════════════════════════════════════════════════════════════
# HIGH KEY / LOW KEY
# ══════════════════════════════════════════════════════════════════

def classify_key(frame_bgr):
    gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum  = float(gray.mean())
    dark_pct  = float((gray < 50).sum()  / gray.size * 100)
    light_pct = float((gray > 200).sum() / gray.size * 100)
    contrast  = float(gray.std())

    if   mean_lum > 160 and dark_pct < 5:
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
        "style":        style,
        "mean_lum":     round(mean_lum, 1),
        "dark_pct":     round(dark_pct, 1),
        "light_pct":    round(light_pct, 1),
        "contrast_std": round(contrast, 1),
    }

# ══════════════════════════════════════════════════════════════════
# CAMERA ANGLE
# ══════════════════════════════════════════════════════════════════

def estimate_camera_angle(frame_bgr):
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    edges  = cv2.Canny(gray, 50, 150)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    horiz_edges  = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horiz_kernel)
    row_sums     = horiz_edges.sum(axis=1)
    horizon_row  = int(row_sums.argmax())
    horizon_pct  = horizon_row / h

    if   horizon_pct < 0.3:          angle = "high angle (camera looking down)"
    elif horizon_pct > 0.7:          angle = "low angle (camera looking up)"
    elif 0.45 < horizon_pct < 0.55:  angle = "eye level"
    else:                            angle = "slight tilt / three-quarter angle"

    diag_mask = np.zeros_like(edges)
    cv2.line(diag_mask, (0,0), (w,h), 255, 20)
    cv2.line(diag_mask, (w,0), (0,h), 255, 20)
    diag_score = float((edges & diag_mask).sum()) / (edges.sum() + 1e-8)
    dutch_tilt = diag_score > 0.35

    center_edges        = edges[h//4:3*h//4, w//4:3*w//4]
    edge_density_center = float(center_edges.mean())
    edge_density_full   = float(edges.mean())
    center_ratio        = edge_density_center / (edge_density_full + 1e-8)

    if   center_ratio > 1.6: shot_type = "close-up (subject fills frame)"
    elif center_ratio > 1.2: shot_type = "medium shot"
    elif center_ratio < 0.8: shot_type = "wide shot (subject small in frame)"
    else:                    shot_type = "medium-wide shot"

    return {
        "angle":       "dutch tilt" if dutch_tilt else angle,
        "shot_type":   shot_type,
        "horizon_pct": round(horizon_pct, 3),
        "dutch_tilt":  dutch_tilt,
        "diag_score":  round(diag_score, 3),
    }

# ══════════════════════════════════════════════════════════════════
# CAMERA MOVEMENT
# ══════════════════════════════════════════════════════════════════

def classify_camera_movement(flow_magnitudes):
    if not flow_magnitudes or len(flow_magnitudes) < 3:
        return {"movement": "unknown", "shake_score": 0.0}
    arr   = np.array(flow_magnitudes)
    mean  = arr.mean()
    std   = arr.std()
    shake = float(std / (mean + 1e-8))
    if   mean < 0.3:                 movement = "static (locked-off)"
    elif shake > 1.2:                movement = "handheld (shaky)"
    elif mean > 2.5 and shake < 0.6: movement = "fast pan / tracking"
    elif mean > 1.0 and shake < 0.5: movement = "slow pan / dolly"
    elif std < 0.3 and mean > 0.5:   movement = "smooth zoom"
    else:                            movement = "mixed / organic movement"
    return {
        "movement":   movement,
        "shake_score": round(shake, 3),
        "mean_flow":  round(float(mean), 3),
    }

# ══════════════════════════════════════════════════════════════════
# ★ SUBJECT LIGHTING — core new feature
# ══════════════════════════════════════════════════════════════════

def analyze_subject_lighting(frame_bgr, bbox: dict):
    """
    Given a bounding box from YOLO, analyze exactly how light
    falls on that subject region.

    bbox keys: x, y, w, h  (top-left origin)

    Returns:
      - light_direction_on_subject : where light hits the subject
      - shadow_side                : opposite side
      - highlight_coords           : brightest pixel region (x,y,w,h)
      - shadow_coords              : darkest pixel region  (x,y,w,h)
      - lit_ratio                  : fraction of subject that is well-lit
      - falloff_direction          : how brightness drops across subject
      - backlit                    : bool
      - rim_light                  : bool
      - exposure_quality           : over/under/good
      - light_type_on_subject      : hard / soft / diffuse
      - color_temp_on_subject      : kelvin estimate for subject region only
      - intensity_map_coords       : 3x3 grid of brightness values + coords
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    fh, fw = frame_bgr.shape[:2]

    # Clamp to frame
    x  = max(0, x);  y  = max(0, y)
    x2 = min(fw, x + w);  y2 = min(fh, y + h)
    if x2 <= x or y2 <= y:
        return None

    roi      = frame_bgr[y:y2, x:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(float)
    rh, rw   = roi_gray.shape

    if rh < 6 or rw < 6:
        return None

    # ── 3×3 intensity grid ──────────────────────────────────────
    grid_vals  = []
    grid_coords = []
    cell_h = rh // 3
    cell_w = rw  // 3
    labels = [
        ["top-left",    "top-center",    "top-right"],
        ["mid-left",    "mid-center",    "mid-right"],
        ["bottom-left", "bottom-center", "bottom-right"],
    ]
    for row in range(3):
        for col in range(3):
            r0 = row * cell_h;  r1 = r0 + cell_h
            c0 = col * cell_w;  c1 = c0 + cell_w
            cell    = roi_gray[r0:r1, c0:c1]
            mean_v  = float(cell.mean())
            grid_vals.append(mean_v)
            grid_coords.append({
                "zone":    labels[row][col],
                "bbox":    {
                    "x": x + c0, "y": y + r0,
                    "w": cell_w, "h": cell_h,
                    "x2": x + c1, "y2": y + r1,
                },
                "mean_brightness": round(mean_v, 1),
            })

    # ── Brightest / darkest zone ─────────────────────────────────
    bright_idx  = int(np.argmax(grid_vals))
    dark_idx    = int(np.argmin(grid_vals))
    bright_zone = grid_coords[bright_idx]
    dark_zone   = grid_coords[dark_idx]

    # ── Highlight pixel coords (top 5% brightest) ────────────────
    thresh_hi   = np.percentile(roi_gray, 95)
    hi_mask     = (roi_gray >= thresh_hi).astype(np.uint8)
    hi_coords_list = np.argwhere(hi_mask)
    if len(hi_coords_list):
        hy0 = int(hi_coords_list[:,0].min()) + y
        hy1 = int(hi_coords_list[:,0].max()) + y
        hx0 = int(hi_coords_list[:,1].min()) + x
        hx1 = int(hi_coords_list[:,1].max()) + x
        highlight_coords = {
            "x": hx0, "y": hy0,
            "w": hx1-hx0, "h": hy1-hy0,
            "x2": hx1, "y2": hy1,
            "brightness_threshold": round(float(thresh_hi), 1),
        }
    else:
        highlight_coords = None

    # ── Shadow pixel coords (bottom 10% darkest) ─────────────────
    thresh_lo  = np.percentile(roi_gray, 10)
    lo_mask    = (roi_gray <= thresh_lo).astype(np.uint8)
    lo_coords_list = np.argwhere(lo_mask)
    if len(lo_coords_list):
        sy0 = int(lo_coords_list[:,0].min()) + y
        sy1 = int(lo_coords_list[:,0].max()) + y
        sx0 = int(lo_coords_list[:,1].min()) + x
        sx1 = int(lo_coords_list[:,1].max()) + x
        shadow_coords = {
            "x": sx0, "y": sy0,
            "w": sx1-sx0, "h": sy1-sy0,
            "x2": sx1, "y2": sy1,
            "brightness_threshold": round(float(thresh_lo), 1),
        }
    else:
        shadow_coords = None

    # ── Light direction on subject ───────────────────────────────
    top_mean    = np.mean(grid_vals[0:3])
    bottom_mean = np.mean(grid_vals[6:9])
    left_mean   = np.mean([grid_vals[0], grid_vals[3], grid_vals[6]])
    right_mean  = np.mean([grid_vals[2], grid_vals[5], grid_vals[8]])
    center_mean = float(grid_vals[4])

    directional = {
        "top": top_mean, "bottom": bottom_mean,
        "left": left_mean, "right": right_mean,
    }
    brightest_side = max(directional, key=directional.get)
    darkest_side   = min(directional, key=directional.get)

    edge_avg = np.mean([top_mean, bottom_mean, left_mean, right_mean])
    backlit  = bool(edge_avg > center_mean * 1.25)
    rim_lit  = bool(
        edge_avg > center_mean * 1.1 and
        float(roi_gray.std()) > 30
    )

    if backlit:
        light_dir = "backlit — light behind subject"
        shadow_side = "front (facing camera)"
    else:
        dir_map = {
            "top":    "top-lit — light falling from above",
            "bottom": "bottom-lit — light from below (uplighting)",
            "left":   "side-lit from subject's right (camera left)",
            "right":  "side-lit from subject's left (camera right)",
        }
        light_dir   = dir_map.get(brightest_side, "front-lit")
        shadow_side = {
            "top": "bottom of subject",
            "bottom": "top of subject",
            "left": "left side of subject",
            "right": "right side of subject",
        }.get(brightest_side, "unknown")

    # ── Falloff direction ────────────────────────────────────────
    lr_diff = left_mean - right_mean
    tb_diff = top_mean  - bottom_mean
    if abs(lr_diff) > abs(tb_diff):
        falloff = "left-to-right" if lr_diff > 0 else "right-to-left"
    else:
        falloff = "top-to-bottom" if tb_diff > 0 else "bottom-to-top"

    # ── Lit ratio ────────────────────────────────────────────────
    well_lit_threshold = 80
    lit_ratio = float((roi_gray > well_lit_threshold).sum() / roi_gray.size)

    # ── Exposure quality ─────────────────────────────────────────
    mean_roi = float(roi_gray.mean())
    if   mean_roi > 210: exposure = "overexposed"
    elif mean_roi < 40:  exposure = "underexposed"
    elif mean_roi > 160: exposure = "slightly bright"
    elif mean_roi < 80:  exposure = "slightly dark"
    else:                exposure = "well exposed"

    # ── Hard vs soft light ───────────────────────────────────────
    # Hard light = sharp shadow edges → high gradient magnitude
    grad_x   = cv2.Sobel(roi_gray.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
    grad_y   = cv2.Sobel(roi_gray.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = float(np.sqrt(grad_x**2 + grad_y**2).mean())
    if   grad_mag > 25: light_type = "hard light (sharp shadows, direct source)"
    elif grad_mag > 12: light_type = "semi-soft light"
    else:               light_type = "soft / diffuse light"

    # ── Color temp on subject only ───────────────────────────────
    ct = estimate_color_temp(roi)

    # ── Reflection falling onto subject from below ───────────────
    bottom_strip = roi_gray[int(rh*0.75):, :]
    top_strip    = roi_gray[:int(rh*0.25), :]
    bounce_light = bool(
        bottom_strip.mean() > top_strip.mean() * 0.85 and
        bottom_strip.mean() > 60
    )

    return {
        "light_direction_on_subject": light_dir,
        "shadow_side":                shadow_side,
        "falloff_direction":          falloff,
        "backlit":                    backlit,
        "rim_light":                  rim_lit,
        "bounce_light_from_below":    bounce_light,
        "lit_ratio":                  round(lit_ratio, 3),
        "exposure_quality":           exposure,
        "light_type_on_subject":      light_type,
        "gradient_magnitude":         round(grad_mag, 2),
        "color_temp_on_subject":      ct,
        "highlight_coords":           highlight_coords,
        "shadow_coords":              shadow_coords,
        "brightest_zone":             bright_zone,
        "darkest_zone":               dark_zone,
        "intensity_map": {
            "grid_3x3": grid_coords,
            "top_mean":    round(top_mean, 1),
            "bottom_mean": round(bottom_mean, 1),
            "left_mean":   round(left_mean, 1),
            "right_mean":  round(right_mean, 1),
            "center_mean": round(center_mean, 1),
        },
    }

# ══════════════════════════════════════════════════════════════════
# FRAME-LEVEL FULL ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyze_frame(frame_bgr, detections=None):
    """
    Full frame cinema analysis.
    If detections (from YOLO) are passed in, also runs
    per-subject lighting analysis with coordinates.
    """
    temp    = estimate_color_temp(frame_bgr)
    source  = classify_light_source(frame_bgr, temp)
    dirn    = estimate_light_direction(frame_bgr)
    reflect = detect_floor_reflection(frame_bgr)
    key     = classify_key(frame_bgr)
    cam     = estimate_camera_angle(frame_bgr)

    result = {
        "color_temperature": temp,
        "light_source":      source,
        "light_direction":   dirn,
        "floor_reflection":  reflect,
        "key_style":         key,
        "camera_angle":      cam,
    }

    # ── Per-subject lighting ──────────────────────────────────────
    if detections:
        subject_lighting = []
        for det in detections:
            bbox    = det.get("bbox", {})
            sl      = analyze_subject_lighting(frame_bgr, bbox)
            if sl:
                subject_lighting.append({
                    "subject_id":    det.get("id"),
                    "label":         det.get("label"),
                    "subject_bbox":  bbox,
                    "lighting":      sl,
                })
        result["subject_lighting"] = subject_lighting

    return result

# ══════════════════════════════════════════════════════════════════
# FULL VIDEO CINEMA ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyze_video_cinema(video_path: str,
                          sample_rate: int = 15,
                          detections_by_frame: dict = None):
    """
    Sample every `sample_rate` frames.
    If detections_by_frame is provided (dict keyed by frame_idx),
    per-subject lighting is computed for each sampled frame.

    Returns per-clip summary + per-segment breakdown.
    """
    cap       = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    sampled   = []
    flow_mags = []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.resize(gray, (224, 224))

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray_r, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            flow_mags.append(float(mag.mean()))

        if frame_idx % sample_rate == 0:
            dets = None
            if detections_by_frame:
                dets = detections_by_frame.get(frame_idx)
            analysis = analyze_frame(frame, detections=dets)
            analysis["timestamp_sec"] = round(frame_idx / fps, 2)
            analysis["frame_idx"]     = frame_idx
            sampled.append(analysis)

        prev_gray = gray_r
        frame_idx += 1

    cap.release()

    if not sampled:
        return None

    # ── Clip-level summary ────────────────────────────────────────
    summary = {
        "dominant_light_source":     most_common(
            [s["light_source"]["source"]      for s in sampled]),
        "dominant_light_direction":  most_common(
            [s["light_direction"]["direction"] for s in sampled]),
        "dominant_key_style":        most_common(
            [s["key_style"]["style"]           for s in sampled]),
        "dominant_camera_angle":     most_common(
            [s["camera_angle"]["angle"]        for s in sampled]),
        "dominant_shot_type":        most_common(
            [s["camera_angle"]["shot_type"]    for s in sampled]),
        "floor_reflection_detected": any(
            s["floor_reflection"]["reflection_detected"] for s in sampled),
        "floor_reflection_type":     most_common(
            [s["floor_reflection"]["reflection_type"]   for s in sampled]),
        "avg_color_temp_kelvin":     safe_mean(
            [s["color_temperature"]["kelvin"]  for s in sampled]),
        "avg_luminance":             safe_mean(
            [s["key_style"]["mean_lum"]        for s in sampled]),
        "avg_contrast":              safe_mean(
            [s["key_style"]["contrast_std"]    for s in sampled]),
        "camera_movement":           classify_camera_movement(flow_mags),
    }

    # ── Per-subject lighting summary across clip ──────────────────
    all_subject_lighting = [
        sl for s in sampled
        for sl in s.get("subject_lighting", [])
    ]
    if all_subject_lighting:
        summary["subject_lighting_summary"] = {
            "dominant_light_direction_on_subjects": most_common(
                [sl["lighting"]["light_direction_on_subject"]
                 for sl in all_subject_lighting]),
            "dominant_light_type": most_common(
                [sl["lighting"]["light_type_on_subject"]
                 for sl in all_subject_lighting]),
            "dominant_exposure": most_common(
                [sl["lighting"]["exposure_quality"]
                 for sl in all_subject_lighting]),
            "backlit_frames_pct": round(
                sum(1 for sl in all_subject_lighting
                    if sl["lighting"]["backlit"])
                / len(all_subject_lighting) * 100, 1),
            "rim_light_frames_pct": round(
                sum(1 for sl in all_subject_lighting
                    if sl["lighting"]["rim_light"])
                / len(all_subject_lighting) * 100, 1),
            "bounce_light_pct": round(
                sum(1 for sl in all_subject_lighting
                    if sl["lighting"]["bounce_light_from_below"])
                / len(all_subject_lighting) * 100, 1),
            "avg_lit_ratio": safe_mean(
                [sl["lighting"]["lit_ratio"]
                 for sl in all_subject_lighting]),
        }

    # ── Per-segment breakdown ─────────────────────────────────────
    seg_size = 5
    segments = []
    for i in range(0, len(sampled), seg_size):
        chunk   = sampled[i:i + seg_size]
        t_start = chunk[0]["timestamp_sec"]
        t_end   = chunk[-1]["timestamp_sec"]
        seg_flows = flow_mags[
            int(t_start * fps): int(t_end * fps)
        ] if flow_mags else []

        seg_subject_lighting = [
            sl for c in chunk
            for sl in c.get("subject_lighting", [])
        ]

        seg = {
            "t_start":         t_start,
            "t_end":           t_end,
            "light_source":    most_common(
                [c["light_source"]["source"]      for c in chunk]),
            "light_direction": most_common(
                [c["light_direction"]["direction"] for c in chunk]),
            "key_style":       most_common(
                [c["key_style"]["style"]           for c in chunk]),
            "camera_angle":    most_common(
                [c["camera_angle"]["angle"]        for c in chunk]),
            "shot_type":       most_common(
                [c["camera_angle"]["shot_type"]    for c in chunk]),
            "reflection":      any(
                c["floor_reflection"]["reflection_detected"]
                for c in chunk),
            "avg_kelvin":      safe_mean(
                [c["color_temperature"]["kelvin"]  for c in chunk]),
            "avg_luminance":   safe_mean(
                [c["key_style"]["mean_lum"]        for c in chunk]),
            "camera_movement": classify_camera_movement(seg_flows)["movement"],
        }

        if seg_subject_lighting:
            seg["subject_lighting"] = [
                {
                    "subject_id":   sl["subject_id"],
                    "label":        sl["label"],
                    "subject_bbox": sl["subject_bbox"],
                    "light_direction_on_subject":
                        sl["lighting"]["light_direction_on_subject"],
                    "shadow_side":
                        sl["lighting"]["shadow_side"],
                    "light_type":
                        sl["lighting"]["light_type_on_subject"],
                    "exposure":
                        sl["lighting"]["exposure_quality"],
                    "backlit":
                        sl["lighting"]["backlit"],
                    "rim_light":
                        sl["lighting"]["rim_light"],
                    "bounce_light":
                        sl["lighting"]["bounce_light_from_below"],
                    "lit_ratio":
                        sl["lighting"]["lit_ratio"],
                    "highlight_coords":
                        sl["lighting"]["highlight_coords"],
                    "shadow_coords":
                        sl["lighting"]["shadow_coords"],
                    "color_temp_on_subject":
                        sl["lighting"]["color_temp_on_subject"],
                    "intensity_map":
                        sl["lighting"]["intensity_map"],
                }
                for sl in seg_subject_lighting
            ]

        segments.append(seg)

    return {
        "summary":       summary,
        "segments":      segments,
        "frame_samples": sampled,
    }