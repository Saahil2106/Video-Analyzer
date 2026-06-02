from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import os, json, threading, subprocess, shutil, cv2, numpy as np
from pathlib import Path
from model              import clf, EFFECTS, get_metadata_status, get_metadata_missing_message
from analyzer           import extract_features, build_metadata
from subject_tracker    import track_subjects
from cinema_analyzer    import analyze_video_cinema
from feature_extractor  import extract_all_features
from reid_tracker       import (POVSwitcher, ReIDModel, IdentityGallery,
                                extract_person_crops, crop_from_bbox,
                                get_reid_model, match_persons_across_cameras)
from json_utils         import to_json_safe

# ── NEW: Kalman + MCT imports ──────────────────────────────────
from tracker.kalman_tracker import KalmanMultiTracker
from tracker.mct            import MultiCameraTracker


app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FFMPEG  = shutil.which("ffmpeg")


# ── Global state ───────────────────────────────────────────────
train_state   = {"status": "idle",    "progress": 0, "result": None, "error": None}
track_state   = {"status": "idle",    "progress": 0,
                 "result": None, "result_file": None}
extract_state = {"status": "idle",    "progress": 0,
                 "result": None, "result_file": None}
reid_state    = {"status": "idle",    "progress": 0,
                 "result": None, "result_file": None,
                 "persons": [], "timeline": []}
pov_state     = {"status": "idle",    "switcher": None,
                 "sources": [],       "index_built": False}

# ── NEW: active MCT instance (persists between requests) ───────
_mct_instance: MultiCameraTracker | None = None


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════


def convert_to_mp4(src_path: str) -> str:
    src = Path(src_path)
    if src.suffix.lower() == ".mp4":
        return src_path
    out_path = str(src.with_suffix(".mp4"))
    if os.path.exists(out_path):
        return out_path
    if not FFMPEG:
        print("  [warn] ffmpeg not found")
        return src_path
    print(f"  Converting {src.name} → MP4...")
    result = subprocess.run([
        FFMPEG, "-y", "-i", src_path,
        "-vcodec", "libx264", "-acodec", "aac",
        "-movflags", "+faststart", "-preset", "fast", "-crf", "23",
        out_path
    ], capture_output=True)
    return out_path if result.returncode == 0 else src_path


def merge_tracking_into_result(result_path: str, tracking_data: dict):
    with open(result_path) as f:
        metadata = json.load(f)

    video_file = (metadata.get("basic_metadata") or {}).get("file") \
                 or metadata.get("file", "")
    src_path   = os.path.join(app.config["UPLOAD_FOLDER"], video_file)

    if not os.path.exists(src_path):
        print(f"  [merge] source video not found: {src_path}")
        return metadata

    detections_by_frame = {}
    for shot in tracking_data.get("shots", []):
        for frame in shot.get("frames", []):
            detections_by_frame[frame["frame"]] = frame["detections"]

    print(f"  Re-running cinema analysis with {len(detections_by_frame)} frames of tracking data...")
    cinema = analyze_video_cinema(
        src_path,
        sample_rate=15,
        detections_by_frame=detections_by_frame,
    )

    if cinema:
        metadata.setdefault("advanced_metadata", {})["cinematography"] = {
            "clip_summary": cinema["summary"],
            "segments":     cinema["segments"],
        }

    metadata.setdefault("advanced_metadata", {})["subject_tracking"] = \
        tracking_data.get("clip_summary", {})

    metadata = to_json_safe(metadata)

    with open(result_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("  Merge complete.")
    return metadata


def _get_first_frame(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _bbox_dict_from_xywh(x, y, w, h) -> dict:
    return {
        "x": int(x), "y": int(y),
        "w": int(w), "h": int(h),
        "x2": int(x + w), "y2": int(y + h),
        "cx": int(x + w // 2), "cy": int(y + h // 2),
    }


# ══════════════════════════════════════════════════════════════
# KALMAN + MCT + YOLO-WORLD + POSE  (plug-and-play)
# ══════════════════════════════════════════════════════════════

def run_yolo_with_kalman_mct(
    video_path: str,
    camera_id:  str = "cam0",
    progress_cb=None,
) -> dict:
    """
    Two-model pipeline:
      1. YOLO-World  — open-vocabulary person detection (no false labels)
      2. YOLO11-pose — skeleton keypoints on confirmed person crops
    Enriched with Kalman tracking + MCT Re-ID per frame.
    Falls back to single YOLO11-pose model if YOLO-World is unavailable.
    """
    global _mct_instance

    from ultralytics import YOLO

    # ── Model 1: YOLO-World (open vocabulary) ─────────────────
    world_model  = None
    USE_WORLD    = False
    try:
        from ultralytics import YOLOWorld
        world_model = YOLOWorld("yolov8x-worldv2.pt")
        world_model.set_classes(["person"])
        USE_WORLD = True
        print("  [tracker] YOLO-World loaded — open vocabulary mode")
    except Exception as e:
        print(f"  [tracker] YOLO-World unavailable ({e})")
        print("            Install: pip install ultralytics")
        print("            Falling back to YOLO11-pose only...")

    # ── Model 2: YOLO11-pose (skeleton) ───────────────────────
    pose_model = None
    for model_name in ("yolo11m-pose.pt", "yolov8m-pose.pt", "yolov8n-pose.pt"):
        try:
            pose_model = YOLO(model_name)
            print(f"  [tracker] pose model loaded: {model_name}")
            break
        except Exception:
            continue

    if pose_model is None and not USE_WORLD:
        raise RuntimeError("No YOLO model could be loaded.")

    # If YOLO-World failed, use pose model for detection too
    if not USE_WORLD:
        world_model = pose_model

    # ── MCT instance ──────────────────────────────────────────
    if _mct_instance is None:
        _mct_instance = MultiCameraTracker(cameras=[camera_id])
    elif camera_id not in _mct_instance.cameras:
        _mct_instance.add_camera(camera_id)

    # ── Video capture ─────────────────────────────────────────
    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_area   = None

    all_frames = []
    frame_idx  = 0
    label_freq = {}
    total_dets = 0
    total_pose = 0

    # ── Detection thresholds ──────────────────────────────────
    CONF_THRESH = 0.40
    SIZE_THRESH = 0.001   # min fraction of frame area
    NMS_IOU     = 0.65

    # ── COCO pose skeleton edges ──────────────────────────────
    COCO_EDGES = [
        (0,1),(0,2),(1,3),(2,4),
        (5,6),
        (5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_area is None:
            frame_area = frame.shape[0] * frame.shape[1]

        raw_dets = []

        # ══════════════════════════════════════════════════════
        # STEP 1 — Detect persons with YOLO-World
        # ══════════════════════════════════════════════════════
        det_results = world_model(frame, conf=CONF_THRESH, iou=NMS_IOU, verbose=False)

        person_boxes = []  # (x1,y1,x2,y2,conf) — feed to pose model

        for r in det_results:
            for box in r.boxes:
                cls_id       = int(box.cls[0])
                label        = world_model.names[cls_id]
                conf         = float(box.conf[0])
                x1,y1,x2,y2 = [float(v) for v in box.xyxy[0].tolist()]
                w_box        = x2 - x1
                h_box        = y2 - y1

                # Only persons — YOLO-World already filtered but be safe
                if "person" not in label.lower():
                    continue
                if conf < CONF_THRESH:
                    continue
                if (w_box * h_box) / frame_area < SIZE_THRESH:
                    continue
                # Aspect ratio gate: persons are taller than wide
                if h_box / w_box < 0.5:
                    continue

                person_boxes.append((x1, y1, x2, y2, conf))

        # ══════════════════════════════════════════════════════
        # STEP 2 — Run pose model on each person crop
        # ══════════════════════════════════════════════════════
        for (x1, y1, x2, y2, conf) in person_boxes:
            w_box = x2 - x1
            h_box = y2 - y1

            det = {
                "label":      "person",
                "confidence": round(conf, 3),
                "bbox": {
                    "x": round(x1, 1), "y": round(y1, 1),
                    "w": round(w_box, 1), "h": round(h_box, 1),
                },
                "position_in_frame": (
                    "left"   if x1 < frame.shape[1] * 0.33 else
                    "right"  if x1 > frame.shape[1] * 0.66 else "center"
                ),
                "relative_size": (
                    "large"  if (w_box * h_box) / frame_area > 0.15 else
                    "medium" if (w_box * h_box) / frame_area > 0.04 else "small"
                ),
                "depth_hint": (
                    "foreground" if (w_box * h_box) / frame_area > 0.10 else
                    "midground"  if (w_box * h_box) / frame_area > 0.03 else "background"
                ),
            }

            # ── Crop person region and run pose model ──────────
            if pose_model is not None and USE_WORLD:
                pad    = 10
                cx1    = max(0, int(x1) - pad)
                cy1    = max(0, int(y1) - pad)
                cx2    = min(frame.shape[1], int(x2) + pad)
                cy2    = min(frame.shape[0], int(y2) + pad)
                crop   = frame[cy1:cy2, cx1:cx2]

                if crop.size > 0:
                    try:
                        pose_results = pose_model(
                            crop, conf=0.35, iou=0.65, verbose=False
                        )
                        # Pick the highest-confidence detection in the crop
                        best_kpts      = None
                        best_kpts_conf = None
                        best_box_conf  = -1.0

                        for pr in pose_results:
                            kpts = pr.keypoints if hasattr(pr, "keypoints") else None
                            for pi, pbox in enumerate(pr.boxes):
                                pc = float(pbox.conf[0])
                                if pc > best_box_conf:
                                    best_box_conf  = pc
                                    best_kpts      = (kpts.xy[pi].tolist()
                                                      if kpts is not None and pi < len(kpts.xy)
                                                      else None)
                                    best_kpts_conf = (kpts.conf[pi].tolist()
                                                      if kpts is not None
                                                         and kpts.conf is not None
                                                         and pi < len(kpts.conf)
                                                      else None)

                        if best_kpts is not None:
                            # Remap crop-relative coords back to full-frame coords
                            kp_xy   = [
                                [kp[0] + cx1, kp[1] + cy1]
                                for kp in best_kpts
                            ]
                            kp_conf = best_kpts_conf or [1.0] * len(kp_xy)

                            visible_kps = [
                                {
                                    "x": round(float(kp[0]), 1),
                                    "y": round(float(kp[1]), 1),
                                    "confidence": round(float(kc), 3),
                                    "visible": float(kc) > 0.4,
                                }
                                for kp, kc in zip(kp_xy, kp_conf)
                            ]

                            skel_edges = []
                            for (a, b) in COCO_EDGES:
                                if a < len(visible_kps) and b < len(visible_kps):
                                    kpa, kpb = visible_kps[a], visible_kps[b]
                                    skel_edges.append({
                                        "x1": kpa["x"], "y1": kpa["y"],
                                        "x2": kpb["x"], "y2": kpb["y"],
                                        "confidence": round(
                                            (kpa["confidence"] + kpb["confidence"]) / 2, 3
                                        ),
                                    })

                            vis_count = sum(1 for kp in visible_kps if kp["visible"])
                            det["pose"] = {
                                "keypoints":      visible_kps,
                                "skeleton_edges": skel_edges,
                                "visible_joints": vis_count,
                                "total_joints":   len(visible_kps),
                            }
                            total_pose += 1

                    except Exception as pose_err:
                        print(f"  [pose] crop inference error: {pose_err}")

            # ── Fallback: single-model mode (no YOLO-World) ────
            elif not USE_WORLD:
                # pose_model IS the detection model — extract kpts directly
                for r in det_results:
                    kpts = r.keypoints if hasattr(r, "keypoints") else None
                    if kpts is None:
                        continue
                    # Match by closest box center
                    cx_det = x1 + w_box / 2
                    cy_det = y1 + h_box / 2
                    best_i, best_d = -1, float("inf")
                    for pi, pbox in enumerate(r.boxes):
                        px1,py1,px2,py2 = pbox.xyxy[0].tolist()
                        d = ((px1+px2)/2 - cx_det)**2 + ((py1+py2)/2 - cy_det)**2
                        if d < best_d:
                            best_d, best_i = d, pi

                    if best_i >= 0 and best_i < len(kpts.xy):
                        kp_xy   = kpts.xy[best_i].tolist()
                        kp_conf = (kpts.conf[best_i].tolist()
                                   if kpts.conf is not None else [1.0]*len(kp_xy))
                        visible_kps = [
                            {
                                "x": round(float(kp[0]), 1),
                                "y": round(float(kp[1]), 1),
                                "confidence": round(float(kc), 3),
                                "visible": float(kc) > 0.4,
                            }
                            for kp, kc in zip(kp_xy, kp_conf)
                        ]
                        skel_edges = []
                        for (a, b) in COCO_EDGES:
                            if a < len(visible_kps) and b < len(visible_kps):
                                kpa, kpb = visible_kps[a], visible_kps[b]
                                skel_edges.append({
                                    "x1": kpa["x"], "y1": kpa["y"],
                                    "x2": kpb["x"], "y2": kpb["y"],
                                    "confidence": round(
                                        (kpa["confidence"] + kpb["confidence"]) / 2, 3
                                    ),
                                })
                        vis_count = sum(1 for kp in visible_kps if kp["visible"])
                        det["pose"] = {
                            "keypoints":      visible_kps,
                            "skeleton_edges": skel_edges,
                            "visible_joints": vis_count,
                            "total_joints":   len(visible_kps),
                        }
                        total_pose += 1
                    break  # only one results object in single-model mode

            raw_dets.append(det)

        # ══════════════════════════════════════════════════════
        # STEP 3 — Kalman + MCT enrichment
        # ══════════════════════════════════════════════════════
        enriched = _mct_instance.update(
            camera_id    = camera_id,
            frame_idx    = frame_idx,
            detections   = raw_dets,
            frame_pixels = frame,
        )

        for d in enriched:
            label_freq[d["label"]] = label_freq.get(d["label"], 0) + 1
            total_dets += 1

        all_frames.append({
            "frame":         frame_idx,
            "timestamp_sec": round(frame_idx / fps, 3),
            "shot_number":   0,
            "subject_count": len(enriched),
            "detections":    enriched,
        })

        frame_idx += 1
        if progress_cb and frame_idx % 10 == 0:
            progress_cb(min(99, int(frame_idx / total_frames * 100)))

    cap.release()
    if progress_cb:
        progress_cb(100)

    mct_summary = _mct_instance.summary()

    return {
        "fps":          fps,
        "total_frames": frame_idx,
        "shots": [{
            "shot_number": 0,
            "start_frame": 0,
            "end_frame":   frame_idx - 1,
            "frames":      all_frames,
        }],
        "clip_summary": {
            "fps":                      fps,
            "total_frames_analyzed":    frame_idx,
            "total_shots":              1,
            "total_detections":         total_dets,
            "total_pose_detections":    total_pose,
            "avg_detections_per_frame": round(total_dets / max(frame_idx, 1), 2),
            "avg_joints_per_pose":      0,
            "label_frequency":          label_freq,
            "mct_total_global_ids":     mct_summary["total_global_ids"],
            "mct_cross_camera_matches": mct_summary["cross_camera_matches"],
            "kalman_active_tracks":     len(
                _mct_instance.kf_trackers[camera_id].get_active_tracks()
            ),
            "model_mode": "yolo-world+pose" if USE_WORLD else "yolo11-pose-only",
        },
        "mct_summary":          mct_summary,
        "cross_camera_matches": _mct_instance.get_cross_camera_matches(),
        "active_tracks":        _mct_instance.kf_trackers[camera_id].get_active_tracks(),
    }

# ══════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

@app.route("/train", methods=["POST"])
def train():
    def run():
        train_state.update({
            "status": "training",
            "progress": 0,
            "result": None,
            "error": None,
        })
        try:
            result = clf.train(progress_cb=lambda p: train_state.update({"progress": p}))
            train_state.update({
                "status": "done",
                "progress": 100,
                "result": to_json_safe(result),
                "error": None,
            })
        except FileNotFoundError:
            train_state.update({
                "status": "error",
                "result": None,
                "error": get_metadata_missing_message(),
            })
            import traceback; traceback.print_exc()
        except Exception as e:
            train_state.update({
                "status": "error",
                "result": None,
                "error": f"Training failed: {e}",
            })
            import traceback; traceback.print_exc()
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"started": True})

@app.route("/train/status")
def train_status():
    return jsonify(to_json_safe(train_state))

@app.route("/metadata/status")
def metadata_status():
    return jsonify(get_metadata_status())


# ══════════════════════════════════════════════════════════════
# ANALYZE
# ══════════════════════════════════════════════════════════════

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f   = request.files["video"]
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    fname    = secure_filename(f.filename)
    src_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    f.save(src_path)

    if not clf.trained:
        return jsonify({"error": "Model not trained yet. Go to the Train tab first."}), 400

    mp4_path = convert_to_mp4(src_path)
    mp4_name = Path(mp4_path).name

    features = extract_features(src_path)
    if features is None:
        return jsonify({"error": "Could not process video — too short or corrupted."}), 400

    predictions = clf.predict(features)
    metadata    = build_metadata(src_path, predictions, features)
    metadata["basic_metadata"]["playable_file"] = mp4_name
    metadata = to_json_safe(metadata)

    result_fname = fname + "_result.json"
    result_path  = os.path.join(app.config["UPLOAD_FOLDER"], result_fname)
    with open(result_path, "w") as out:
        json.dump(metadata, out, indent=2)

    tracking_path = src_path + "_tracking.json"
    if os.path.exists(tracking_path):
        print("  Found existing tracking data — merging...")
        with open(tracking_path) as tf:
            tracking_data = json.load(tf)
        metadata = merge_tracking_into_result(result_path, tracking_data)

    return jsonify({"metadata": metadata, "result_file": result_fname})


# ══════════════════════════════════════════════════════════════
# TRACK — Kalman + MCT + YOLO, then merges into analysis result
# ══════════════════════════════════════════════════════════════

@app.route("/track", methods=["POST"])
def track():
    global track_state, _mct_instance

    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f    = request.files["video"]
    ext  = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    fname    = secure_filename(f.filename)
    src_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)

    if not os.path.exists(src_path):
        f.save(src_path)
    else:
        f.stream.read()

    camera_id   = Path(fname).stem
    track_state = {"status": "running", "progress": 0,
                   "result": None, "result_file": None}

    def run_tracking():
        global track_state, _mct_instance
        try:
            def cb(p):
                track_state["progress"] = p

            # Primary: Kalman + MCT pipeline
            try:
                result = run_yolo_with_kalman_mct(
                    video_path  = src_path,
                    camera_id   = camera_id,
                    progress_cb = cb,
                )
                print(f"  [tracker] Kalman+MCT done. "
                      f"global_ids={result['mct_summary']['total_global_ids']}  "
                      f"cross_cam={result['mct_summary']['cross_camera_matches']}")

            except Exception as kalman_err:
                # Fallback: original subject_tracker
                print(f"  [tracker] Kalman/MCT failed ({kalman_err}), "
                      f"falling back to subject_tracker...")
                _mct_instance = None
                result = track_subjects(src_path, progress_cb=cb)

            result      = to_json_safe(result)
            out_path    = src_path + "_tracking.json"
            track_fname = fname + "_tracking.json"

            with open(out_path, "w") as out:
                json.dump(result, out, indent=2)

            result_path = os.path.join(
                app.config["UPLOAD_FOLDER"], fname + "_result.json"
            )
            merged_metadata = None
            if os.path.exists(result_path):
                print("  Analysis result found — merging tracking data...")
                merged_metadata = merge_tracking_into_result(result_path, result)

            track_state.update({
                "status":          "done",
                "progress":        100,
                "result":          result,
                "result_file":     track_fname,
                "merged_metadata": merged_metadata,
                "mct_summary":     result.get("mct_summary"),
                "active_tracks":   result.get("active_tracks", []),
            })

        except Exception as e:
            track_state["status"] = "error"
            track_state["error"]  = str(e)
            print(f"Tracking error: {e}")
            import traceback; traceback.print_exc()

    threading.Thread(target=run_tracking, daemon=True).start()
    return jsonify({"started": True})


@app.route("/track/status")
def track_status():
    return jsonify(to_json_safe({
        "status":          track_state["status"],
        "progress":        track_state["progress"],
        "result":          track_state["result"],
        "result_file":     track_state.get("result_file"),
        "merged_metadata": track_state.get("merged_metadata"),
        "mct_summary":     track_state.get("mct_summary"),
        "active_tracks":   track_state.get("active_tracks", []),
        "error":           track_state.get("error"),
    }))


# ══════════════════════════════════════════════════════════════
# NEW: MCT STATUS — inspect cross-camera gallery at any time
# ══════════════════════════════════════════════════════════════

@app.route("/mct/status")
def mct_status():
    if _mct_instance is None:
        return jsonify({"status": "no_instance",
                        "message": "Run /track first to initialise the MCT."})
    return jsonify(to_json_safe({
        "status":               "active",
        "summary":              _mct_instance.summary(),
        "cross_camera_matches": _mct_instance.get_cross_camera_matches(),
        "cameras":              _mct_instance.cameras,
        "gallery_size":         len(_mct_instance.gallery.entries),
    }))

@app.route("/mct/reset", methods=["POST"])
def mct_reset():
    """Clear the MCT gallery before a new multi-camera session."""
    global _mct_instance
    _mct_instance = None
    return jsonify({"reset": True})


# ══════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════

@app.route("/report/<path:filename>")
def report(filename):
    safe = secure_filename(filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], safe)

    if not os.path.exists(path):
        alt = os.path.join(app.config["UPLOAD_FOLDER"], safe + "_result.json")
        if os.path.exists(alt):
            path = alt
        else:
            available = [f for f in os.listdir(app.config["UPLOAD_FOLDER"])
                         if f.endswith("_result.json")]
            return (f"Report not found: {safe}<br>"
                    f"Available: {available}"), 404

    with open(path) as f:
        data = json.load(f)
    return render_template("report.html", data=data, filename=safe)


# ══════════════════════════════════════════════════════════════
# VIDEO SERVING  (byte-range for seeking)
# ══════════════════════════════════════════════════════════════

@app.route("/video/<path:filename>")
def serve_video(filename):
    safe = secure_filename(filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], safe)

    if not os.path.exists(path):
        mp4_alt = os.path.join(app.config["UPLOAD_FOLDER"],
                               Path(safe).stem + ".mp4")
        if os.path.exists(mp4_alt):
            path = mp4_alt
        else:
            return jsonify({"error": f"Video not found: {safe}"}), 404

    file_size    = os.path.getsize(path)
    ext          = Path(path).suffix.lower()
    mime         = ("video/mp4"       if ext == ".mp4"  else
                    "video/webm"      if ext == ".webm" else
                    "video/x-msvideo" if ext == ".avi"  else
                    "video/mp4")
    range_header = request.headers.get("Range")

    if not range_header:
        return send_file(path, mimetype=mime)

    match  = range_header.replace("bytes=", "").split("-")
    byte1  = int(match[0])
    byte2  = int(match[1]) if match[1] else min(byte1 + 1024*1024, file_size - 1)
    length = byte2 - byte1 + 1

    with open(path, "rb") as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, 206, mimetype=mime, direct_passthrough=True)
    rv.headers.add("Content-Range",  f"bytes {byte1}-{byte2}/{file_size}")
    rv.headers.add("Accept-Ranges",  "bytes")
    rv.headers.add("Content-Length", str(length))
    return rv


# ══════════════════════════════════════════════════════════════
# DOWNLOAD
# ══════════════════════════════════════════════════════════════

@app.route("/download/<path:filename>")
def download(filename):
    safe = secure_filename(filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], safe)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

@app.route("/extract", methods=["POST"])
def extract():
    global extract_state

    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f   = request.files["video"]
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    fname    = secure_filename(f.filename)
    src_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)

    if not os.path.exists(src_path):
        f.save(src_path)
    else:
        f.stream.read()

    extract_state = {"status": "running", "progress": 0,
                     "result": None, "result_file": None}

    def run():
        global extract_state
        try:
            def cb(p):
                extract_state["progress"] = p
            result = extract_all_features(src_path, progress_cb=cb)
            result = to_json_safe(result)
            out_fname = fname + "_features.json"
            out_path  = os.path.join(app.config["UPLOAD_FOLDER"], out_fname)
            with open(out_path, "w") as out:
                json.dump(result, out, indent=2)
            extract_state.update({
                "status":      "done",
                "progress":    100,
                "result":      result,
                "result_file": out_fname,
            })
        except Exception as e:
            extract_state["status"] = "error"
            extract_state["error"]  = str(e)
            print(f"Feature extraction error: {e}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"started": True})

@app.route("/extract/status")
def extract_status():
    return jsonify(to_json_safe({
        "status":      extract_state["status"],
        "progress":    extract_state["progress"],
        "result":      extract_state["result"],
        "result_file": extract_state.get("result_file"),
        "error":       extract_state.get("error"),
    }))


# ══════════════════════════════════════════════════════════════
# RE-ID — build identity index, shares MCT gallery with /track
# ══════════════════════════════════════════════════════════════

@app.route("/reid", methods=["POST"])
def reid():
    global reid_state, pov_state, _mct_instance

    data  = request.get_json(force=True, silent=True) or {}
    files = data.get("files", [])

    if not files:
        return jsonify({"error": "No files provided. Pass { 'files': ['cam1.mp4', ...] }"}), 400

    paths = []
    for fname in files:
        safe = secure_filename(fname)
        path = os.path.join(app.config["UPLOAD_FOLDER"], safe)
        if not os.path.exists(path):
            return jsonify({"error": f"File not found in uploads: {safe}"}), 404
        paths.append(path)

    reid_state = {"status": "running", "progress": 0,
                  "result": None, "result_file": None,
                  "persons": [], "timeline": []}
    pov_state  = {"status": "building", "switcher": None,
                  "sources": paths,     "index_built": False}

    camera_ids    = [Path(p).stem for p in paths]
    _mct_instance = MultiCameraTracker(cameras=camera_ids)

    def run_reid():
        global reid_state, pov_state, _mct_instance
        try:
            switcher = POVSwitcher(
                camera_paths=paths,
                similarity_threshold=0.70,
            )

            def cb(pct):
                reid_state["progress"] = pct

            switcher.build_index(sample_rate=15, progress_cb=cb)

            # Run Kalman+MCT over every camera
            print("  [reid] Running Kalman+MCT over all cameras...")
            total_cams = len(paths)
            for ci, (cam_path, cam_id) in enumerate(zip(paths, camera_ids)):
                base_pct = 80 + int(ci / total_cams * 15)
                try:
                    run_yolo_with_kalman_mct(
                        video_path  = cam_path,
                        camera_id   = cam_id,
                        progress_cb = lambda p: reid_state.update({
                            "progress": base_pct + int(p * 0.15 / 100)
                        }),
                    )
                    print(f"    MCT cam {cam_id} done.")
                except Exception as e:
                    print(f"    [warn] MCT failed for {cam_id}: {e}")

            pov_state["switcher"]    = switcher
            pov_state["index_built"] = True
            pov_state["status"]      = "ready"

            persons     = switcher.gallery.known_ids()
            out_fname   = "reid_index.json"
            out_path    = os.path.join(app.config["UPLOAD_FOLDER"], out_fname)
            mct_summary = _mct_instance.summary() if _mct_instance else {}
            summary     = to_json_safe({
                "cameras":              files,
                "persons":              persons,
                "mct_summary":          mct_summary,
                "cross_camera_matches": (_mct_instance.get_cross_camera_matches()
                                         if _mct_instance else []),
            })

            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)

            reid_state.update({
                "status":      "done",
                "progress":    100,
                "persons":     persons,
                "result_file": out_fname,
                "result":      summary,
                "mct_summary": mct_summary,
            })
            print(f"  Re-ID done. Persons: {persons}  "
                  f"MCT global_ids: {mct_summary.get('total_global_ids',0)}")

        except Exception as e:
            reid_state["status"] = "error"
            reid_state["error"]  = str(e)
            pov_state["status"]  = "error"
            print(f"Re-ID error: {e}")
            import traceback; traceback.print_exc()

    threading.Thread(target=run_reid, daemon=True).start()
    return jsonify({"started": True, "cameras": files})

@app.route("/reid/status")
def reid_status():
    return jsonify(to_json_safe({
        "status":      reid_state["status"],
        "progress":    reid_state["progress"],
        "persons":     reid_state.get("persons", []),
        "result":      reid_state.get("result"),
        "result_file": reid_state.get("result_file"),
        "mct_summary": reid_state.get("mct_summary"),
        "error":       reid_state.get("error"),
    }))


# ══════════════════════════════════════════════════════════════
# POV SWITCH
# ══════════════════════════════════════════════════════════════

@app.route("/pov/switch", methods=["POST"])
def pov_switch():
    if not pov_state.get("index_built"):
        return jsonify({"error": "Re-ID index not built yet. Call /reid first."}), 400

    switcher: POVSwitcher = pov_state["switcher"]
    data      = request.get_json(force=True, silent=True) or {}
    person_id = data.get("person_id")

    if person_id is None:
        bbox_data   = data.get("bbox")
        camera_file = data.get("camera_file")

        if not bbox_data or not camera_file:
            return jsonify({
                "error": "Provide either 'person_id' or both 'bbox' and 'camera_file'."
            }), 400

        safe     = secure_filename(camera_file)
        cam_path = os.path.join(app.config["UPLOAD_FOLDER"], safe)
        if not os.path.exists(cam_path):
            return jsonify({"error": f"Camera file not found: {safe}"}), 404

        frame = _get_first_frame(cam_path)
        if frame is None:
            return jsonify({"error": "Could not read first frame of camera file."}), 400

        bbox      = _bbox_dict_from_xywh(
            bbox_data.get("x", 0), bbox_data.get("y", 0),
            bbox_data.get("w", 50), bbox_data.get("h", 100),
        )
        crop      = crop_from_bbox(frame, bbox)
        emb       = get_reid_model().extract_embedding(crop)
        person_id = switcher.gallery.assign(emb)
        print(f"  Clicked person resolved to person_id={person_id}")

        if _mct_instance:
            cam_id = Path(camera_file).stem
            gid    = _mct_instance.gallery.match_or_register(
                embedding=emb, label="person",
                camera_id=cam_id, local_track_id=person_id, frame_idx=0,
            )
            print(f"  MCT global_id: {gid}")

    print(f"  Generating POV timeline for person_id={person_id}...")
    timeline = switcher.generate_pov_timeline(target_person_id=person_id)

    clean_timeline = [{
        "frame_index":  e["frame_index"],
        "camera_index": e["camera_index"],
        "camera_file":  Path(e["camera_path"]).name,
        "bbox":         e["bbox"],
        "similarity":   e["similarity"],
    } for e in timeline]

    out = to_json_safe({
        "person_id": person_id,
        "cameras":   [Path(p).name for p in pov_state["sources"]],
        "timeline":  clean_timeline,
        "switches":  _count_switches(clean_timeline),
    })

    out_path = os.path.join(app.config["UPLOAD_FOLDER"], "pov_timeline.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"  Timeline: {len(clean_timeline)} entries, {out['switches']} camera switches.")
    return jsonify(out)


def _count_switches(timeline: list) -> int:
    switches, prev = 0, None
    for entry in timeline:
        if prev is not None and entry["camera_index"] != prev:
            switches += 1
        prev = entry["camera_index"]
    return switches


# ══════════════════════════════════════════════════════════════
# POV LIVE SELECT
# ══════════════════════════════════════════════════════════════

@app.route("/pov/select", methods=["POST"])
def pov_select():
    data        = request.get_json(force=True, silent=True) or {}
    camera_file = data.get("camera_file")
    start_frame = int(data.get("start_frame", 0))

    if not camera_file:
        return jsonify({"error": "Provide 'camera_file'."}), 400

    safe     = secure_filename(camera_file)
    cam_path = os.path.join(app.config["UPLOAD_FOLDER"], safe)
    if not os.path.exists(cam_path):
        return jsonify({"error": f"Camera file not found: {safe}"}), 404

    result_bbox = _live_bbox_select(cam_path, start_frame)

    if result_bbox is None:
        return jsonify({"cancelled": True})

    return jsonify({"bbox": result_bbox, "camera_file": camera_file})


def _live_bbox_select(video_path: str, start_frame: int = 0) -> dict | None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    WIN = "Live Playback  |  SPACE=pause  |  drag=select  |  ENTER=confirm  |  Q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    mouse = {"drawing": False, "start": None, "end": None, "done": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse.update({"drawing": True, "start": (x, y),
                          "end": (x, y), "done": False})
        elif event == cv2.EVENT_MOUSEMOVE and mouse["drawing"]:
            mouse["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse.update({"drawing": False, "end": (x, y), "done": True})

    cv2.setMouseCallback(WIN, on_mouse)

    paused, frozen_frame, result_bbox = False, None, None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                ret, frame = cap.read()
                if not ret:
                    break

        display = (frozen_frame if paused else frame).copy()

        if paused and mouse["start"] and mouse["end"]:
            cv2.rectangle(display, mouse["start"], mouse["end"], (0, 255, 0), 2)
            for pt in [mouse["start"], mouse["end"]]:
                cv2.circle(display, pt, 4, (0, 255, 0), -1)

        h_frame = display.shape[0]
        if paused:
            status = ("Box drawn — ENTER to confirm | SPACE to redraw | Q to cancel"
                      if mouse["done"] else
                      "PAUSED — drag a box around the target | SPACE to resume | Q to cancel")
            colour = (0, 255, 0) if mouse["done"] else (0, 200, 255)
        else:
            status = "Playing — SPACE to pause and select | Q to quit"
            colour = (255, 255, 255)

        cv2.rectangle(display, (0, h_frame - 35),
                      (display.shape[1], h_frame), (0, 0, 0), -1)
        cv2.putText(display, status, (10, h_frame - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        cv2.imshow(WIN, display)
        wait_ms = max(1, int(1000 / fps)) if not paused else 20
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord(" "):
            if not paused:
                paused       = True
                frozen_frame = frame.copy()
                mouse.update({"start": None, "end": None, "done": False})
            else:
                paused = False
                mouse.update({"start": None, "end": None, "done": False})

        elif key in (13, 10):
            if paused and mouse["done"] and mouse["start"] and mouse["end"]:
                sx, sy = mouse["start"]
                ex, ey = mouse["end"]
                x, y   = min(sx, ex), min(sy, ey)
                w, h   = abs(ex - sx), abs(ey - sy)
                if w > 5 and h > 5:
                    result_bbox = _bbox_dict_from_xywh(x, y, w, h)
                    break

        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return result_bbox


# ══════════════════════════════════════════════════════════════
# CROSS-CAMERA MATCH
# ══════════════════════════════════════════════════════════════

@app.route("/pov/match", methods=["POST"])
def pov_match():
    from subject_tracker import get_det_model, get_pose_model, detect_frame

    data      = request.get_json(force=True, silent=True) or {}
    file_a    = data.get("file_a")
    file_b    = data.get("file_b")
    frame_a_n = int(data.get("frame_a", 0))
    frame_b_n = int(data.get("frame_b", 0))
    threshold = float(data.get("threshold", 0.75))

    if not file_a or not file_b:
        return jsonify({"error": "Provide 'file_a' and 'file_b'."}), 400

    path_a = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file_a))
    path_b = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file_b))

    for p in [path_a, path_b]:
        if not os.path.exists(p):
            return jsonify({"error": f"File not found: {Path(p).name}"}), 404

    def read_frame(path, frame_n):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    frame_a = read_frame(path_a, frame_a_n)
    frame_b = read_frame(path_b, frame_b_n)

    if frame_a is None or frame_b is None:
        return jsonify({"error": "Could not read frames from one or both files."}), 400

    det_model  = get_det_model()
    pose_model = get_pose_model()
    dets_a     = detect_frame(frame_a, det_model, pose_model)
    dets_b     = detect_frame(frame_b, det_model, pose_model)

    matches = match_persons_across_cameras(
        frame_a, dets_a, frame_b, dets_b, threshold=threshold,
    )

    # Enrich with MCT global IDs if available
    if _mct_instance:
        cam_a = Path(file_a).stem
        cam_b = Path(file_b).stem
        for m in matches:
            m["global_id_a"] = _mct_instance.get_global_id(cam_a, m.get("track_id_a"))
            m["global_id_b"] = _mct_instance.get_global_id(cam_b, m.get("track_id_b"))

    return jsonify(to_json_safe({
        "file_a":          file_a,
        "file_b":          file_b,
        "frame_a":         frame_a_n,
        "frame_b":         frame_b_n,
        "threshold":       threshold,
        "matches":         matches,
        "total_persons_a": len([d for d in dets_a if d["label"] == "person"]),
        "total_persons_b": len([d for d in dets_b if d["label"] == "person"]),
    }))


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("tracker", exist_ok=True)
    init_file = Path("tracker/__init__.py")
    if not init_file.exists():
        init_file.write_text("")
    print("\n" + "="*50)
    print("  EffectScan — Video Analysis Platform")
    print("  http://localhost:5000")
    if not FFMPEG:
        print("\n  [!] ffmpeg not found — AVI/MOV won't play in browser")
        print("      Fix: winget install ffmpeg")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, threaded=True)
