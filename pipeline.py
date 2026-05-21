"""
pipeline.py
───────────
Single unified entry point for the entire video analysis pipeline.

Wraps subject_tracker + feature_extractor + reid_tracker into one
clean interface. No hardcoding — everything is driven by a config
dict or keyword arguments.

Quick start
───────────
    from pipeline import Pipeline

    # Single video — full analysis
    p = Pipeline("myvideo.mp4")
    results = p.run()

    # Single video — pause playback, draw box, track that object
    p = Pipeline("myvideo.mp4")
    results = p.run(interactive=True)

    # Multiple cameras — auto POV switching, user picks target live
    p = Pipeline(["cam1.mp4", "cam2.mp4", "cam3.mp4"])
    results = p.run(interactive=True)

Controls (interactive mode)
───────────────────────────
    SPACE       → pause / resume playback
    drag mouse  → draw bounding box while paused
    ENTER       → confirm selection
    Q / ESC     → quit without selection
"""

import cv2
import numpy as np
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════

class Pipeline:
    """
    Unified pipeline that auto-detects what needs to run based on
    how many video sources are provided.

    Single video  → subject tracking + feature extraction
    Multi video   → above + Re-ID + POV switching timeline

    Parameters
    ──────────
    sources : str | list[str]
        One video path or a list of video paths (one per camera).

    sample_rate : int (default 15)
        How often to sample frames for feature extraction + Re-ID.

    similarity_threshold : float (default 0.75)
        Cosine similarity required to match the same person across cameras.

    sot_backend : str (default "csrt")
        Single-object tracker backend: "csrt", "kcf", or "mosse".

    progress : bool (default True)
        Print progress to console.
    """

    def __init__(
        self,
        sources,
        sample_rate: int = 15,
        similarity_threshold: float = 0.75,
        sot_backend: str = "csrt",
        progress: bool = True,
    ):
        if isinstance(sources, (str, Path)):
            self.sources = [str(sources)]
        else:
            self.sources = [str(s) for s in sources]

        self.sample_rate          = sample_rate
        self.similarity_threshold = similarity_threshold
        self.sot_backend          = sot_backend
        self.progress             = progress
        self.is_multi_camera      = len(self.sources) > 1

        self._switcher    = None
        self._index_built = False

        self._log(f"Pipeline initialised — "
                  f"{'multi-camera' if self.is_multi_camera else 'single video'} mode")
        self._log(f"Sources: {self.sources}")

    # ══════════════════════════════════════════════════════════════
    # LOGGING
    # ══════════════════════════════════════════════════════════════

    def _log(self, msg: str):
        if self.progress:
            print(f"[Pipeline] {msg}")

    def _progress_cb(self, label: str):
        def cb(pct):
            if self.progress:
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r  {label}: [{bar}] {pct}%", end="", flush=True)
            if pct == 100 and self.progress:
                print()
        return cb

    # ══════════════════════════════════════════════════════════════
    # SOURCE VALIDATION
    # ══════════════════════════════════════════════════════════════

    def _validate_sources(self):
        for path in self.sources:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video not found: {path}")
        self._log("All sources validated ✓")

    # ══════════════════════════════════════════════════════════════
    # STEP 1 — Feature Extraction
    # ══════════════════════════════════════════════════════════════

    def extract_features(self, source_index: int = 0) -> dict:
        """
        Run full feature extraction on one video source.
        Returns the complete feature dict (motion, color, brightness,
        texture, scene, objects).
        """
        from feature_extractor import extract_all_features

        path = self.sources[source_index]
        self._log(f"Extracting features from: {Path(path).name}")
        result = extract_all_features(
            path,
            progress_cb=self._progress_cb(f"Features [{Path(path).name}]")
        )
        return result

    # ══════════════════════════════════════════════════════════════
    # STEP 2 — Subject Tracking
    # ══════════════════════════════════════════════════════════════

    def track_subjects(self, source_index: int = 0) -> dict:
        """
        Run YOLO11m + pose tracking on every frame of one video source.
        Returns shot-structured output with per-frame detections.
        """
        from subject_tracker import track_subjects

        path = self.sources[source_index]
        self._log(f"Tracking subjects in: {Path(path).name}")
        result = track_subjects(
            path,
            progress_cb=self._progress_cb(f"Tracking [{Path(path).name}]")
        )
        return result

    # ══════════════════════════════════════════════════════════════
    # STEP 3 — Single-Object Tracking (SOT)
    # ══════════════════════════════════════════════════════════════

    def track_single_object(
        self,
        initial_bbox: tuple,
        source_index: int = 0,
    ) -> dict:
        """
        Lock onto one bounding box and follow it through the video.

        Parameters
        ──────────
        initial_bbox : (x, y, w, h) from pick_bbox_from_playback()
                       or pick_bbox_interactively()
        source_index : which camera/source to use (default 0)

        Returns
        ───────
        { backend, total_frames, found_frames, lost_frames, trajectory }
        """
        from subject_tracker import track_single_object

        path = self.sources[source_index]
        self._log(f"SOT [{self.sot_backend}] on: {Path(path).name}")
        self._log(f"  Initial bbox: {initial_bbox}")

        result = track_single_object(
            path,
            initial_bbox=initial_bbox,
            backend=self.sot_backend,
            progress_cb=self._progress_cb("SOT"),
        )
        self._log(f"  Found: {result['found_frames']} frames, "
                  f"Lost: {result['lost_frames']} frames")
        return result

    # ══════════════════════════════════════════════════════════════
    # STEP 4 — Re-ID Index (multi-camera only)
    # ══════════════════════════════════════════════════════════════

    def build_reid_index(self) -> "Pipeline":
        """
        Build the Re-ID identity index across all cameras.
        Must be called before any POV switching methods.
        Returns self so you can chain: p.build_reid_index().switch_to(1)
        """
        if not self.is_multi_camera:
            self._log("Skipping Re-ID index — only one source provided.")
            return self

        from reid_tracker import POVSwitcher

        self._log("Building Re-ID index across all cameras...")
        self._switcher = POVSwitcher(
            camera_paths=self.sources,
            similarity_threshold=self.similarity_threshold,
        )
        self._switcher.build_index(
            sample_rate=self.sample_rate,
            progress_cb=self._progress_cb("Re-ID Index"),
        )
        self._index_built = True
        self._log(f"Known identities: {self._switcher.gallery.known_ids()}")
        return self

    # ══════════════════════════════════════════════════════════════
    # STEP 5 — POV Switching
    # ══════════════════════════════════════════════════════════════

    def list_persons(self) -> list[int]:
        """
        Return all person IDs detected across all cameras.
        Call after build_reid_index().
        """
        self._require_index()
        ids = self._switcher.gallery.known_ids()
        self._log(f"Detected person IDs: {ids}")
        return ids

    def switch_to(self, person_id: int) -> list[dict]:
        """
        Generate a full frame-by-frame POV timeline that follows
        the given person_id across all cameras.

        Returns list of:
        { frame_index, camera_index, camera_path, bbox, similarity }
        """
        self._require_index()
        self._log(f"Generating POV timeline for person_id={person_id}...")
        timeline = self._switcher.generate_pov_timeline(
            target_person_id=person_id
        )
        self._log(f"Timeline: {len(timeline)} entries across "
                  f"{len(set(e['camera_index'] for e in timeline))} cameras")
        return timeline

    def switch_to_by_crop(self, frame_bgr: np.ndarray,
                           bbox: dict) -> list[dict]:
        """
        Resolve a person from a clicked frame region to their Re-ID
        and return the full POV timeline for that person.

        Parameters
        ──────────
        frame_bgr : the frame the user selected on (BGR numpy array)
        bbox      : dict with x, y, w, h, x2, y2, cx, cy

        Returns same as switch_to().
        """
        self._require_index()
        from reid_tracker import crop_from_bbox, get_reid_model

        crop      = crop_from_bbox(frame_bgr, bbox)
        emb       = get_reid_model().extract_embedding(crop)
        person_id = self._switcher.gallery.assign(emb)

        self._log(f"Selected person resolved to person_id={person_id}")
        return self.switch_to(person_id)

    def best_camera_at(self, frame_index: int,
                        person_id: int | None = None) -> dict | None:
        """
        Return the best camera for a specific frame index.
        Useful for real-time querying without generating the full timeline.
        """
        self._require_index()
        if person_id is not None:
            self._switcher.set_target(person_id)
        return self._switcher.best_camera_for_frame(frame_index)

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════

    def get_first_frame(self, source_index: int = 0) -> np.ndarray:
        """Return the first frame of a video as a BGR numpy array."""
        cap = cv2.VideoCapture(self.sources[source_index])
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(
                f"Cannot read first frame of {self.sources[source_index]}"
            )
        return frame

    def pick_bbox_interactively(self, source_index: int = 0) -> tuple:
        """
        Static fallback: open a single paused frame and let the user
        draw a bounding box using cv2.selectROI.
        Returns (x, y, w, h).

        Use pick_bbox_from_playback() instead for live video selection.
        """
        frame = self.get_first_frame(source_index)
        self._log("Draw a box around the target, then press ENTER/SPACE.")
        bbox = cv2.selectROI(
            "Select target — ENTER to confirm, C to cancel",
            frame,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyAllWindows()
        self._log(f"Selected bbox: {bbox}")
        return bbox   # (x, y, w, h)

    def pick_bbox_from_playback(
        self,
        source_index: int = 0,
        start_frame: int = 0,
    ) -> dict | None:
        """
        Play the video live. User presses SPACE to pause at any frame,
        drags a bounding box around the target, then presses ENTER to
        confirm. Playback resumes if they press SPACE again without drawing.

        Controls
        ────────
        SPACE           → pause / resume playback
        drag mouse      → draw bounding box (only while paused)
        ENTER           → confirm selection and close window
        Q / ESC         → cancel — returns None

        Returns
        ───────
        bbox dict { x, y, w, h, x2, y2, cx, cy }
        or None if cancelled.
        """
        path = self.sources[source_index]
        cap  = cv2.VideoCapture(path)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        WIN = "Live Playback  |  SPACE = pause  |  drag = select  |  ENTER = confirm  |  Q = quit"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 720)

        # ── Mouse state ───────────────────────────────────────────
        mouse = {
            "drawing": False,
            "start":   None,
            "end":     None,
            "done":    False,
        }

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse["drawing"] = True
                mouse["start"]   = (x, y)
                mouse["end"]     = (x, y)
                mouse["done"]    = False
            elif event == cv2.EVENT_MOUSEMOVE and mouse["drawing"]:
                mouse["end"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                mouse["drawing"] = False
                mouse["end"]     = (x, y)
                mouse["done"]    = True

        cv2.setMouseCallback(WIN, on_mouse)

        paused        = False
        frozen_frame  = None
        result_bbox   = None

        self._log(f"Playback started: {Path(path).name}")
        self._log("Press SPACE to pause on the frame you want, "
                  "then drag a box around the target.")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Loop back to start if video ends before selection
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    ret, frame = cap.read()
                    if not ret:
                        self._log("Could not read video — exiting selection.")
                        break

            display = (frozen_frame if paused else frame).copy()

            # ── Overlay: selection rectangle ──────────────────────
            if paused and mouse["start"] and mouse["end"]:
                cv2.rectangle(
                    display,
                    mouse["start"],
                    mouse["end"],
                    (0, 255, 0), 2
                )
                # Corner handles for clarity
                for pt in [mouse["start"], mouse["end"]]:
                    cv2.circle(display, pt, 4, (0, 255, 0), -1)

            # ── Overlay: status bar ───────────────────────────────
            h_frame = display.shape[0]
            if paused:
                if mouse["done"]:
                    status = "Box drawn — press ENTER to confirm | SPACE to redraw | Q to cancel"
                    colour = (0, 255, 0)
                else:
                    status = "PAUSED — drag a box around the target | SPACE to resume | Q to cancel"
                    colour = (0, 200, 255)
            else:
                status = "Playing — SPACE to pause and select | Q to quit"
                colour = (255, 255, 255)

            cv2.rectangle(display, (0, h_frame - 35),
                          (display.shape[1], h_frame), (0, 0, 0), -1)
            cv2.putText(display, status,
                        (10, h_frame - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

            cv2.imshow(WIN, display)

            wait_ms = max(1, int(1000 / fps)) if not paused else 20
            key = cv2.waitKey(wait_ms) & 0xFF

            # ── SPACE — pause / resume ────────────────────────────
            if key == ord(" "):
                if not paused:
                    paused       = True
                    frozen_frame = frame.copy()
                    mouse["start"] = mouse["end"] = None
                    mouse["done"]  = False
                    self._log("Paused. Drag a box around the target.")
                else:
                    paused = False
                    mouse["start"] = mouse["end"] = None
                    mouse["done"]  = False
                    self._log("Resumed.")

            # ── ENTER — confirm selection ─────────────────────────
            elif key in (13, 10):   # ENTER (CR or LF)
                if paused and mouse["done"] and mouse["start"] and mouse["end"]:
                    sx, sy = mouse["start"]
                    ex, ey = mouse["end"]
                    x = min(sx, ex)
                    y = min(sy, ey)
                    w = abs(ex - sx)
                    h = abs(ey - sy)
                    if w > 5 and h > 5:
                        result_bbox = {
                            "x":  x,       "y":  y,
                            "w":  w,       "h":  h,
                            "x2": x + w,   "y2": y + h,
                            "cx": x + w // 2,
                            "cy": y + h // 2,
                        }
                        self._log(f"Selection confirmed: {result_bbox}")
                        break
                    else:
                        self._log("Box too small — drag a larger area.")
                elif not paused:
                    self._log("Press SPACE to pause first, then draw a box.")
                else:
                    self._log("Draw a box first, then press ENTER.")

            # ── Q / ESC — cancel ──────────────────────────────────
            elif key in (ord("q"), 27):
                self._log("Selection cancelled.")
                break

        cap.release()
        cv2.destroyAllWindows()
        return result_bbox

    def _require_index(self):
        if not self._index_built or self._switcher is None:
            raise RuntimeError(
                "Re-ID index not built yet. Call build_reid_index() first."
            )

    # ══════════════════════════════════════════════════════════════
    # ONE-CALL FULL RUN
    # ══════════════════════════════════════════════════════════════

    def run(
        self,
        target_person_id: int | None = None,
        target_bbox: dict | None = None,
        run_features: bool = True,
        run_tracking: bool = True,
        run_sot_bbox: tuple | None = None,
        interactive: bool = False,
    ) -> dict:
        """
        Run the full pipeline in one call.

        Parameters
        ──────────
        target_person_id : follow this person ID across cameras (optional)

        target_bbox      : bbox dict — skip interactive selection and use
                           this box directly (optional)

        run_features     : extract visual features — motion, color,
                           brightness, texture, scene, objects (default True)

        run_tracking     : run MOT + Kalman subject tracking (default True)

        run_sot_bbox     : (x, y, w, h) tuple — run SOT on this specific
                           bounding box (optional, non-interactive)

        interactive      : open live playback window so the user can pause
                           and draw a box on any frame (default False).
                           When True:
                             • Sets run_sot_bbox automatically from selection
                             • In multi-camera mode, resolves the selected
                               person across all cameras for POV switching
                           Overrides target_bbox and run_sot_bbox if set.

        Returns
        ───────
        {
          "sources"       : list of source paths,
          "selected_bbox" : bbox dict if interactive/target_bbox was used,
          "features"      : { source_path: feature_dict },   if run_features
          "tracking"      : { source_path: tracking_dict },  if run_tracking
          "sot"           : sot_result_dict,                 if SOT was run
          "timeline"      : pov_timeline_list,               if multi-camera
          "persons"       : list of detected person IDs,     if multi-camera
        }
        """
        self._validate_sources()
        results = {"sources": self.sources}

        # ── Interactive live selection ────────────────────────────
        # Opens playback window — user pauses, draws box, presses ENTER.
        # This runs BEFORE any analysis so the selection is ready.
        if interactive:
            self._log("─── Interactive Selection ────────────────────")
            self._log("Opening live playback for target selection...")
            selected = self.pick_bbox_from_playback(source_index=0)

            if selected is not None:
                target_bbox  = selected
                run_sot_bbox = (
                    selected["x"], selected["y"],
                    selected["w"], selected["h"],
                )
                results["selected_bbox"] = selected
                self._log(f"Target selected — bbox: {selected}")
            else:
                self._log("No selection made — continuing without target.")

        elif target_bbox is not None:
            # Non-interactive bbox provided directly
            results["selected_bbox"] = target_bbox
            if run_sot_bbox is None:
                run_sot_bbox = (
                    target_bbox["x"], target_bbox["y"],
                    target_bbox["w"], target_bbox["h"],
                )

        # ── Feature extraction ────────────────────────────────────
        if run_features:
            self._log("─── Feature Extraction ───────────────────────")
            results["features"] = {}
            for i, src in enumerate(self.sources):
                results["features"][src] = self.extract_features(i)

        # ── Subject tracking (MOT + Kalman) ───────────────────────
        if run_tracking:
            self._log("─── Subject Tracking ─────────────────────────")
            results["tracking"] = {}
            for i, src in enumerate(self.sources):
                results["tracking"][src] = self.track_subjects(i)

        # ── Single-object tracking ────────────────────────────────
        if run_sot_bbox is not None:
            self._log("─── Single-Object Tracking ───────────────────")
            results["sot"] = self.track_single_object(
                initial_bbox=run_sot_bbox,
                source_index=0,
            )

        # ── Re-ID + POV switching (multi-camera only) ─────────────
        if self.is_multi_camera:
            self._log("─── Re-ID Index ──────────────────────────────")
            self.build_reid_index()
            results["persons"] = self.list_persons()

            if target_bbox is not None:
                # Resolve the selected bbox to a person identity
                self._log("─── POV Switch (from selection) ──────────")
                frame = self.get_first_frame(0)
                results["timeline"] = self.switch_to_by_crop(frame, target_bbox)

            elif target_person_id is not None:
                self._log(f"─── POV Switch (person_id={target_person_id}) ─")
                results["timeline"] = self.switch_to(target_person_id)

            else:
                # Default: follow the first detected person
                self._log("─── POV Switch (default person_id=1) ────────")
                results["timeline"] = self.switch_to(1)

        self._log("Pipeline complete ✓")
        return results