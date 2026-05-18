import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter



# ── Model registry ────────────────────────────────────────────────
# Person detector      → YOLO-World restricted to "person"
# Object detector      → YOLO-World restricted to selected equipment labels
# Pose estimator       → YOLO11m-pose for 17-keypoint human pose estimation
from ultralytics import YOLO, YOLOWorld



_person_det_model = None
_object_det_model = None
_pose_model = None



def get_person_det_model():
    global _person_det_model
    if _person_det_model is None:
        print("Loading YOLO-World person detection model...")
        _person_det_model = YOLOWorld("yolov8x-worldv2.pt")
        _person_det_model.set_classes(["person"])
    return _person_det_model



def get_object_det_model():
    global _object_det_model
    if _object_det_model is None:
        print("Loading YOLO-World object detection model...")
        _object_det_model = YOLOWorld("yolov8x-worldv2.pt")
        _object_det_model.set_classes([
            "archery bow",
            "recurve bow",
            "longbow",
            "compound bow",
            "bow weapon",
            "arrow"
        ])
    return _object_det_model



def get_det_model():
    # Backward-compatible alias expected elsewhere in the app
    return get_person_det_model()



def get_pose_model():
    global _pose_model
    if _pose_model is None:
        print("Loading YOLO11m-pose model...")
        _pose_model = YOLO("yolo11m-pose.pt")
    return _pose_model



# ── COCO keypoint skeleton connections (17 keypoints) ─────────────
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]



KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]



# ── Tracking classes ──────────────────────────────────────────────
ALWAYS_TRACK_CLASSES = {
    "person",
}


ALLOWED_OBJECT_LABELS = {
    "archery bow",
    "recurve bow",
    "longbow",
    "compound bow",
    "bow weapon",
    "arrow",
}



# ── Confidence thresholds ─────────────────────────────────────────
CONF_THRESHOLD_PERSON = 0.10
CONF_THRESHOLD_OBJECT = 0.01
CONF_THRESHOLD_DEFAULT = 0.60



# ── Minimum bounding-box area as fraction of frame ────────────────
MIN_AREA_FRACTION = {
    "person": 0.0012,
    "archery bow": 0.00015,
    "recurve bow": 0.00015,
    "longbow": 0.00015,
    "compound bow": 0.00015,
    "bow weapon": 0.00015,
    "arrow": 0.00005,
    "default": 0.03,
}



# ── Temporal gate ─────────────────────────────────────────────────
MIN_FRAMES_SEEN = {
    "person": 1,
    "archery bow": 1,
    "recurve bow": 1,
    "longbow": 1,
    "compound bow": 1,
    "bow weapon": 1,
    "arrow": 1,
    "default": 2,
}



# ══════════════════════════════════════════════════════════════════
# KALMAN FILTER TRACKER
# ══════════════════════════════════════════════════════════════════



class KalmanTracker:
    """
    Per-object Kalman filter that smooths bounding-box centre positions
    and predicts where an object will be on the next sampled frame.

    State vector : [cx, cy, vx, vy]
    Measurement  : [cx, cy]
    """

    def __init__(self, cx: float, cy: float):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array(
            [[cx], [cy], [0.0], [0.0]], np.float32)
        self.missed = 0
        self.track_id = id(self)

    def predict(self) -> tuple:
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def update(self, cx: float, cy: float):
        self.kf.correct(np.array([[cx], [cy]], np.float32))
        self.missed = 0

    def mark_missed(self):
        self.missed += 1



_KALMAN_MAX_MISSED = 8
_KALMAN_MATCH_DIST_PERSON = 120
_KALMAN_MATCH_DIST_OBJECT = 140



def _match_detection_to_tracker(cx: float, cy: float,
                                trackers: dict,
                                used_keys: set | None = None,
                                max_dist: float = 120.0) -> str | None:
    """
    Find the closest existing tracker within the given distance threshold.
    Returns the tracker key or None if no match.
    """
    best_key = None
    best_dist = max_dist
    used_keys = used_keys or set()

    for key, kt in trackers.items():
        if key in used_keys:
            continue
        pcx, pcy = kt.predict()
        dist = float(np.hypot(cx - pcx, cy - pcy))
        if dist < best_dist:
            best_dist = dist
            best_key = key
    return best_key



def _new_track_key(prefix: str, next_track_id: int) -> str:
    return f"{prefix}_{next_track_id}"



# ══════════════════════════════════════════════════════════════════
# SINGLE-OBJECT TRACKER (SOT)
# ══════════════════════════════════════════════════════════════════



def _make_tracker(name: str):
    creators = {
        "csrt":  ["cv2.TrackerCSRT_create", "cv2.legacy.TrackerCSRT_create"],
        "kcf":   ["cv2.TrackerKCF_create", "cv2.legacy.TrackerKCF_create"],
        "mosse": ["cv2.legacy.TrackerMOSSE_create", "cv2.TrackerMOSSE_create"],
    }
    for attr_path in creators.get(name, []):
        try:
            obj = cv2
            for part in attr_path.split(".")[1:]:
                obj = getattr(obj, part)
            return obj()
        except AttributeError:
            continue
    raise RuntimeError(
        f"Could not create tracker '{name}'. "
        f"Install opencv-contrib-python: pip install opencv-contrib-python"
    )



SOT_BACKENDS = {
    "csrt":  lambda: _make_tracker("csrt"),
    "kcf":   lambda: _make_tracker("kcf"),
    "mosse": lambda: _make_tracker("mosse"),
}



def track_single_object(
    video_path: str,
    initial_bbox: tuple,
    backend: str = "csrt",
    progress_cb=None
) -> dict:
    if backend not in SOT_BACKENDS:
        raise ValueError(f"Unknown SOT backend '{backend}'. "
                         f"Choose from: {list(SOT_BACKENDS.keys())}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return {"error": "Could not read first frame", "trajectory": []}

    tracker = SOT_BACKENDS[backend]()
    tracker.init(first_frame, initial_bbox)

    ix, iy, iw, ih = initial_bbox
    kalman = KalmanTracker(ix + iw / 2, iy + ih / 2)

    trajectory = []
    frame_idx = 1
    found_count = 0
    lost_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        found, raw_bbox = tracker.update(frame)
        ts = round(frame_idx / fps, 2)

        if found:
            rx, ry, rw, rh = [int(v) for v in raw_bbox]
            cx = rx + rw // 2
            cy = ry + rh // 2
            kalman.update(float(cx), float(cy))
            scx, scy = kalman.predict()
            bbox_out = {
                "x": rx, "y": ry, "w": rw, "h": rh,
                "x2": rx + rw, "y2": ry + rh,
                "cx": cx, "cy": cy
            }
            found_count += 1
        else:
            scx, scy = kalman.predict()
            kalman.mark_missed()
            bbox_out = None
            lost_count += 1

        trajectory.append({
            "frame": frame_idx,
            "timestamp_sec": ts,
            "found": bool(found),
            "bbox": bbox_out,
            "smoothed_cx": round(scx, 1),
            "smoothed_cy": round(scy, 1),
        })

        if progress_cb and frame_idx % 15 == 0:
            progress_cb(int(frame_idx / max(total_frames, 1) * 100))

        frame_idx += 1

    cap.release()
    if progress_cb:
        progress_cb(100)

    return {
        "backend": backend,
        "total_frames": frame_idx,
        "found_frames": found_count,
        "lost_frames": lost_count,
        "trajectory": trajectory,
    }



# ══════════════════════════════════════════════════════════════════
# SMALL FAST-MOVING OBJECT DETECTOR (background subtraction)
# ══════════════════════════════════════════════════════════════════



def detect_small_moving_objects(prev_gray, curr_gray, fw, fh,
                                label="moving_object",
                                min_area_frac=0.0003,
                                max_area_frac=0.015):
    frame_area = fw * fh
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area_frac * frame_area < area < max_area_frac * frame_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        detections.append({
            "label": label,
            "confidence": 0.5,
            "bbox": {
                "x": x, "y": y, "w": w, "h": h,
                "x2": x + w, "y2": y + h,
                "cx": cx, "cy": cy
            },
            "position_in_frame": get_frame_position(cx, cy, fw, fh),
            "relative_size": get_relative_size(w * h, frame_area),
            "depth_hint": get_depth_hint(w * h, frame_area, cy, fh),
            "pose": None,
        })
    return detections



# ══════════════════════════════════════════════════════════════════
# FRAME POSITION / SIZE / DEPTH HELPERS
# ══════════════════════════════════════════════════════════════════



def get_frame_position(cx, cy, fw, fh):
    col = "left" if cx < fw / 3 else ("right" if cx > 2 * fw / 3 else "center")
    row = "top" if cy < fh / 3 else ("bottom" if cy > 2 * fh / 3 else "center")
    if row == "center" and col == "center":
        return "center"
    if row == "center":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"



def get_relative_size(bbox_area, frame_area):
    ratio = bbox_area / (frame_area + 1e-8)
    if ratio > 0.4:
        return "dominant"
    elif ratio > 0.15:
        return "large"
    elif ratio > 0.05:
        return "medium"
    elif ratio > 0.01:
        return "small"
    else:
        return "tiny"



def get_depth_hint(bbox_area, frame_area, cy, fh):
    size_ratio = bbox_area / (frame_area + 1e-8)
    if size_ratio > 0.15 or cy > 0.6 * fh:
        return "foreground"
    elif size_ratio < 0.02:
        return "background"
    else:
        return "midground"



# ══════════════════════════════════════════════════════════════════
# SHOT BOUNDARY DETECTION
# ══════════════════════════════════════════════════════════════════



def detect_shot_boundaries(video_path: str, threshold: float = 30.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    prev_hist = None
    frame_idx = 0
    boundaries = [0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR))
            if diff > threshold:
                boundaries.append(frame_idx)

        prev_hist = hist
        frame_idx += 1

    boundaries.append(frame_idx)
    cap.release()

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
    return shots, fps



# ══════════════════════════════════════════════════════════════════
# POSE HELPERS
# ══════════════════════════════════════════════════════════════════



def extract_pose_for_detection(pose_results, bbox, conf_threshold=0.2):
    if pose_results is None or pose_results.keypoints is None:
        return None

    kpts_data = pose_results.keypoints.data
    boxes_data = pose_results.boxes

    if kpts_data is None or len(kpts_data) == 0:
        return None

    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    best_iou = 0.2
    best_kpts = None

    for idx in range(len(kpts_data)):
        if boxes_data is not None and idx < len(boxes_data.xyxy):
            px1, py1, px2, py2 = map(float, boxes_data.xyxy[idx].tolist())
            ix1 = max(bx, px1)
            iy1 = max(by, py1)
            ix2 = min(bx + bw, px2)
            iy2 = min(by + bh, py2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = bw * bh + (px2 - px1) * (py2 - py1) - inter
            iou = inter / (union + 1e-8)
        else:
            iou = 1.0 if idx == 0 else 0.0

        if iou > best_iou:
            best_iou = iou
            best_kpts = kpts_data[idx]

    if best_kpts is None:
        return None

    keypoints = []
    for ki, kp in enumerate(best_kpts):
        kx, ky, kc = float(kp[0]), float(kp[1]), float(kp[2])
        keypoints.append({
            "name": KEYPOINT_NAMES[ki],
            "index": ki,
            "x": round(kx, 1),
            "y": round(ky, 1),
            "confidence": round(kc, 3),
            "visible": bool(kc >= conf_threshold and kx > 0 and ky > 0),
        })
    return keypoints



def build_skeleton_edges(keypoints):
    if not keypoints:
        return []
    kp_map = {kp["index"]: kp for kp in keypoints}
    edges = []
    for a, b in SKELETON_CONNECTIONS:
        ka = kp_map.get(a)
        kb = kp_map.get(b)
        if ka and kb and ka["visible"] and kb["visible"]:
            edges.append({
                "x1": ka["x"], "y1": ka["y"],
                "x2": kb["x"], "y2": kb["y"],
                "confidence": round((ka["confidence"] + kb["confidence"]) / 2, 3),
            })
    return edges



# ══════════════════════════════════════════════════════════════════
# PER-FRAME DETECTION (person + object + pose + Kalman smoothing)
# ══════════════════════════════════════════════════════════════════



def detect_frame(frame_bgr,
                 person_det_model,
                 object_det_model,
                 pose_model,
                 conf_threshold=0.25,
                 kalman_trackers: dict | None = None,
                 object_kalman_trackers: dict | None = None,
                 next_track_id: int = 1,
                 next_object_track_id: int = 1):
    """
    1. Run YOLO-World person detector.
    2. Run YOLO-World object detector for equipment.
    3. Run YOLO11m-pose for humans.
    4. Attach Kalman-smoothed tracking for persons and objects separately.
    """
    fh, fw = frame_bgr.shape[:2]
    frame_area = fw * fh

    person_results = person_det_model(frame_bgr, verbose=False, conf=CONF_THRESHOLD_PERSON)[0]
    object_results = object_det_model(frame_bgr, verbose=False, conf=CONF_THRESHOLD_OBJECT)[0]

    try:
        pose_results = pose_model(frame_bgr, verbose=False, conf=0.10)[0]
    except Exception:
        pose_results = None

    detections = []
    seen_person_keys_this_frame = set()
    seen_object_keys_this_frame = set()

    # ── Person detections ───────────────────────────────────────
    for i, box in enumerate(person_results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = person_det_model.names[cls]

        if label != "person":
            continue

        if conf < CONF_THRESHOLD_PERSON:
            continue

        w = x2 - x1
        h = y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2

        area_frac = (w * h) / (frame_area + 1e-8)
        min_frac = MIN_AREA_FRACTION.get("person", MIN_AREA_FRACTION["default"])
        if area_frac < min_frac:
            continue

        bbox = {
            "x": x1, "y": y1, "w": w, "h": h,
            "x2": x2, "y2": y2, "cx": cx, "cy": cy
        }

        det = {
            "id": i + 1,
            "track_id": None,
            "label": label,
            "confidence": round(conf, 3),
            "bbox": bbox,
            "position_in_frame": get_frame_position(cx, cy, fw, fh),
            "relative_size": get_relative_size(w * h, frame_area),
            "depth_hint": get_depth_hint(w * h, frame_area, cy, fh),
            "frame_dimensions": {"width": fw, "height": fh},
            "pose": None,
            "smoothed_cx": None,
            "smoothed_cy": None,
        }

        if kalman_trackers is not None:
            matched_key = _match_detection_to_tracker(
                cx, cy,
                kalman_trackers,
                used_keys=seen_person_keys_this_frame,
                max_dist=_KALMAN_MATCH_DIST_PERSON
            )

            if matched_key is None:
                matched_key = _new_track_key("person", next_track_id)
                kalman_trackers[matched_key] = KalmanTracker(float(cx), float(cy))
                next_track_id += 1
            else:
                kalman_trackers[matched_key].update(float(cx), float(cy))

            scx, scy = kalman_trackers[matched_key].predict()
            det["track_id"] = matched_key
            det["smoothed_cx"] = round(scx, 1)
            det["smoothed_cy"] = round(scy, 1)
            seen_person_keys_this_frame.add(matched_key)

        if pose_results is not None:
            keypoints = extract_pose_for_detection(pose_results, bbox)
            if keypoints:
                det["pose"] = {
                    "keypoints": keypoints,
                    "skeleton_edges": build_skeleton_edges(keypoints),
                    "visible_joints": sum(1 for kp in keypoints if kp["visible"]),
                    "total_joints": len(keypoints),
                }

        detections.append(det)

    # ── Object detections ───────────────────────────────────────
    person_boxes = [d["bbox"] for d in detections if d["label"] == "person"]

    for j, box in enumerate(object_results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = object_det_model.names[cls]

        if label not in ALLOWED_OBJECT_LABELS:
            continue

        if conf < CONF_THRESHOLD_OBJECT:
            continue

        w = x2 - x1
        h = y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2

        area_frac = (w * h) / (frame_area + 1e-8)
        min_frac = MIN_AREA_FRACTION.get(label, 0.00005)
        if area_frac < min_frac:
            continue

        bbox = {
            "x": x1, "y": y1, "w": w, "h": h,
            "x2": x2, "y2": y2, "cx": cx, "cy": cy
        }

        # Prefer objects near a detected person to suppress stray false positives
        near_person = False
        for pb in person_boxes:
            px1, py1, px2, py2 = pb["x"], pb["y"], pb["x2"], pb["y2"]
            expand_x = int((px2 - px1) * 0.9)
            expand_y = int((py2 - py1) * 0.9)
            ex1, ey1 = px1 - expand_x, py1 - expand_y
            ex2, ey2 = px2 + expand_x, py2 + expand_y
            if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                near_person = True
                break

        if person_boxes and not near_person:
            continue

        det = {
            "id": 1000 + j + 1,
            "track_id": None,
            "label": label,
            "confidence": round(conf, 3),
            "bbox": bbox,
            "position_in_frame": get_frame_position(cx, cy, fw, fh),
            "relative_size": get_relative_size(w * h, frame_area),
            "depth_hint": get_depth_hint(w * h, frame_area, cy, fh),
            "frame_dimensions": {"width": fw, "height": fh},
            "pose": None,
            "smoothed_cx": None,
            "smoothed_cy": None,
        }

        if object_kalman_trackers is not None:
            matched_key = _match_detection_to_tracker(
                cx, cy,
                object_kalman_trackers,
                used_keys=seen_object_keys_this_frame,
                max_dist=_KALMAN_MATCH_DIST_OBJECT
            )

            if matched_key is None:
                matched_key = _new_track_key("object", next_object_track_id)
                object_kalman_trackers[matched_key] = KalmanTracker(float(cx), float(cy))
                next_object_track_id += 1
            else:
                object_kalman_trackers[matched_key].update(float(cx), float(cy))

            scx, scy = object_kalman_trackers[matched_key].predict()
            det["track_id"] = matched_key
            det["smoothed_cx"] = round(scx, 1)
            det["smoothed_cy"] = round(scy, 1)
            seen_object_keys_this_frame.add(matched_key)

        detections.append(det)

    # Mark missing person trackers
    if kalman_trackers is not None:
        for key, kt in list(kalman_trackers.items()):
            if key not in seen_person_keys_this_frame:
                kt.mark_missed()
                if kt.missed > _KALMAN_MAX_MISSED:
                    del kalman_trackers[key]

    # Mark missing object trackers
    if object_kalman_trackers is not None:
        for key, kt in list(object_kalman_trackers.items()):
            if key not in seen_object_keys_this_frame:
                kt.mark_missed()
                if kt.missed > _KALMAN_MAX_MISSED:
                    del object_kalman_trackers[key]

    return detections, next_track_id, next_object_track_id



# ══════════════════════════════════════════════════════════════════
# FULL VIDEO SUBJECT TRACKING (MOT + Kalman + Fix 3)
# ══════════════════════════════════════════════════════════════════



def track_subjects(video_path: str, conf_threshold: float = 0.25,
                   progress_cb=None):
    """
    Run person detection + object detection + pose on every frame.
    Person and object tracking are maintained separately across the video.
    """
    person_det_model = get_person_det_model()
    object_det_model = get_object_det_model()
    pose_model = get_pose_model()
    shots, fps = detect_shot_boundaries(video_path)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    shot_frames = defaultdict(list)

    kalman_trackers: dict = {}
    object_kalman_trackers: dict = {}
    next_track_id = 1
    next_object_track_id = 1

    print(f"  Detecting persons + objects + pose in {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, next_track_id, next_object_track_id = detect_frame(
            frame,
            person_det_model,
            object_det_model,
            pose_model,
            conf_threshold,
            kalman_trackers=kalman_trackers,
            object_kalman_trackers=object_kalman_trackers,
            next_track_id=next_track_id,
            next_object_track_id=next_object_track_id
        )
        ts = round(frame_idx / fps, 4)

        shot_num = 1
        for shot in shots:
            if shot["start_frame"] <= frame_idx <= shot["end_frame"]:
                shot_num = shot["shot_number"]
                break

        shot_frames[shot_num].append({
            "frame": frame_idx,
            "timestamp_sec": ts,
            "detections": detections,
            "subject_count": len(detections),
            "labels_present": list(set(d["label"] for d in detections)),
        })

        if progress_cb and frame_idx % 15 == 0:
            progress_cb(int(frame_idx / max(total_frames, 1) * 100))

        frame_idx += 1

    cap.release()

    # ── Shot-level summaries ──────────────────────────────────────
    shots_output = []
    for shot in shots:
        sn = shot["shot_number"]
        frames = shot_frames[sn]
        if not frames:
            continue

        all_labels = [l for f in frames for l in f["labels_present"]]
        label_counts = Counter(all_labels)

        filtered_label_counts = Counter({
            label: count
            for label, count in label_counts.items()
            if count >= MIN_FRAMES_SEEN.get(label, MIN_FRAMES_SEEN["default"])
        })

        dominant_labels = [l for l, _ in filtered_label_counts.most_common(5)]
        avg_subjects = round(
            sum(f["subject_count"] for f in frames) / len(frames), 1
        )

        top_label = dominant_labels[0] if dominant_labels else None
        positions = [
            d["position_in_frame"]
            for f in frames for d in f["detections"]
            if d["label"] == top_label
        ]
        dominant_position = (
            Counter(positions).most_common(1)[0][0]
            if positions else "unknown"
        )

        pose_frames = [f for f in frames if any(d["pose"] for d in f["detections"])]
        pose_coverage = round(len(pose_frames) / max(len(frames), 1) * 100, 1)

        shots_output.append({
            **shot,
            "avg_subject_count": avg_subjects,
            "dominant_subjects": dominant_labels,
            "dominant_position": dominant_position,
            "label_frequency": dict(filtered_label_counts.most_common()),
            "pose_coverage_pct": pose_coverage,
            "frames": frames,
        })

    # ── Clip-level summary ────────────────────────────────────────
    all_detections = [d for s in shots_output for f in s["frames"]
                      for d in f["detections"]]
    all_labels_clip = [d["label"] for d in all_detections]
    clip_label_freq = Counter(all_labels_clip)

    pose_dets = [d for d in all_detections if d["pose"] is not None]
    avg_visible_joints = (
        round(np.mean([d["pose"]["visible_joints"] for d in pose_dets]), 1)
        if pose_dets else 0
    )

    clip_summary = {
        "total_frames_analyzed": frame_idx,
        "total_shots": len(shots_output),
        "total_detections": len(all_detections),
        "unique_labels": list(clip_label_freq.keys()),
        "label_frequency": dict(clip_label_freq.most_common()),
        "avg_detections_per_frame": round(
            len(all_detections) / max(frame_idx, 1), 2
        ),
        "pose_detections": len(pose_dets),
        "avg_visible_joints": avg_visible_joints,
        "fps": round(fps, 2),
    }

    return {"clip_summary": clip_summary, "shots": shots_output}