"""
reid_tracker.py
───────────────
Multi-Camera Tracking (MCT) via appearance-based Re-Identification.

Uses a lightweight OSNet model (via torchreid) to extract a 512-dim
embedding vector from any person crop, then matches identities across
different camera feeds / POV clips using cosine similarity.

Workflow
────────
1. Extract person crops from each camera using subject_tracker.detect_frame()
2. Compute an embedding for each crop via ReIDModel.extract_embedding()
3. Match identities across cameras via match_identities()
4. Switch / merge POVs based on matched identity assignments

Dependencies
────────────
    pip install torchreid torch torchvision

If torchreid is unavailable the module falls back to a histogram-based
embedding that requires only OpenCV — accuracy is lower but functional.
"""

import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
# TORCHREID IMPORT (with graceful fallback)
# ══════════════════════════════════════════════════════════════════

try:
    import torch
    import torchvision.transforms as T
    import torchreid
    _TORCHREID_AVAILABLE = True
except ImportError:
    _TORCHREID_AVAILABLE = False
    print(
        "[reid_tracker] torchreid not found — falling back to histogram embeddings.\n"
        "  For best accuracy: pip install torchreid torch torchvision"
    )


# ══════════════════════════════════════════════════════════════════
# RE-ID MODEL
# ══════════════════════════════════════════════════════════════════

class ReIDModel:
    """
    Wraps OSNet-x0.25 (torchreid) or a histogram fallback.

    Usage
    ─────
    model = ReIDModel()
    emb   = model.extract_embedding(crop_bgr)   # np.ndarray shape (D,)
    """

    # Standard ImageNet normalisation used by OSNet
    _TRANSFORM = None

    def __init__(self, model_name: str = "osnet_x0_25"):
        self.model_name = model_name
        self.model      = None
        self.device     = None
        self._build()

    def _build(self):
        if not _TORCHREID_AVAILABLE:
            print("[ReIDModel] Using histogram fallback (no torchreid).")
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ReIDModel] Loading {self.model_name} on {self.device}...")

        self.model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=1000,
            pretrained=True,
        )
        self.model.eval()
        self.model.to(self.device)

        ReIDModel._TRANSFORM = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),          # standard Re-ID input size
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        print("[ReIDModel] Ready.")

    def extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Compute a normalised embedding vector for a single person crop.

        Parameters
        ──────────
        crop_bgr : BGR image array (H, W, 3), any size — resized internally.

        Returns
        ───────
        np.ndarray of shape (D,) — L2-normalised, float32.
        D = 512 for OSNet, 768 for histogram fallback.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(512, dtype=np.float32)

        if not _TORCHREID_AVAILABLE or self.model is None:
            return self._histogram_embedding(crop_bgr)

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor   = ReIDModel._TRANSFORM(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor)

        emb_np = emb.squeeze().cpu().numpy().astype(np.float32)
        return self._l2_normalize(emb_np)

    def extract_embeddings_batch(self,
                                  crops: list[np.ndarray]) -> list[np.ndarray]:
        """
        Compute embeddings for a list of crops.
        More efficient than calling extract_embedding() in a loop when
        torchreid is available (batched GPU inference).
        """
        if not crops:
            return []

        if not _TORCHREID_AVAILABLE or self.model is None:
            return [self._histogram_embedding(c) for c in crops]

        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                tensors.append(torch.zeros(3, 256, 128))
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(ReIDModel._TRANSFORM(crop_rgb))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embs = self.model(batch)

        return [self._l2_normalize(e.cpu().numpy().astype(np.float32))
                for e in embs]

    # ── Fallback: colour + gradient histogram ─────────────────────
    @staticmethod
    def _histogram_embedding(crop_bgr: np.ndarray) -> np.ndarray:
        """
        256-bin HSV histogram (H + S + V channels) + 256-bin gradient
        magnitude histogram = 768-dim descriptor.
        No deep learning required.
        """
        resized = cv2.resize(crop_bgr, (64, 128))
        hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        h_hist = cv2.calcHist([hsv], [0], None, [256], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()

        gray  = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx    = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy    = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag   = np.sqrt(gx**2 + gy**2).flatten()
        g_hist, _ = np.histogram(mag, bins=256, range=(0, 512))
        g_hist    = g_hist.astype(np.float32)

        emb = np.concatenate([h_hist, s_hist, v_hist, g_hist])
        return ReIDModel._l2_normalize(emb)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)


# ── Singleton accessor (mirrors get_det_model() pattern) ──────────
_reid_model = None

def get_reid_model() -> ReIDModel:
    global _reid_model
    if _reid_model is None:
        _reid_model = ReIDModel()
    return _reid_model


# ══════════════════════════════════════════════════════════════════
# CROP EXTRACTION
# ══════════════════════════════════════════════════════════════════

def crop_from_bbox(frame_bgr: np.ndarray, bbox: dict,
                   padding: float = 0.05) -> np.ndarray:
    """
    Extract a person crop from a frame using a detection bbox dict
    (same format as subject_tracker.detect_frame() output).

    padding : fractional padding added around the bbox (default 5 %).
    """
    fh, fw = frame_bgr.shape[:2]
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0,  x - pad_x)
    y1 = max(0,  y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)

    return frame_bgr[y1:y2, x1:x2].copy()


def extract_person_crops(frame_bgr: np.ndarray,
                          detections: list) -> list[dict]:
    """
    Given a frame and its detection list (from detect_frame()),
    return crops for all 'person' detections.

    Returns list of:
        { "detection_id", "bbox", "crop" }
    """
    crops = []
    for det in detections:
        if det["label"] != "person":
            continue
        crop = crop_from_bbox(frame_bgr, det["bbox"])
        if crop.size == 0:
            continue
        crops.append({
            "detection_id": det["id"],
            "bbox":         det["bbox"],
            "crop":         crop,
        })
    return crops


# ══════════════════════════════════════════════════════════════════
# IDENTITY MATCHING
# ══════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))   # already normalised → dot = cosine


def match_identities(
    embeddings_a: list[np.ndarray],
    embeddings_b: list[np.ndarray],
    threshold: float = 0.75,
) -> list[dict]:
    """
    Greedy nearest-neighbour matching between two sets of embeddings.

    Each embedding in set A is matched to its closest embedding in B
    (if similarity ≥ threshold). One-to-one: once B[j] is matched it
    cannot be matched again.

    Parameters
    ──────────
    embeddings_a  : list of embedding vectors from camera / POV A
    embeddings_b  : list of embedding vectors from camera / POV B
    threshold     : minimum cosine similarity to accept a match (0–1)

    Returns
    ───────
    List of match dicts:
        { "idx_a", "idx_b", "similarity" }
    Unmatched indices from A or B are not included.
    """
    if not embeddings_a or not embeddings_b:
        return []

    # Build full similarity matrix
    sim_matrix = np.array([
        [cosine_similarity(ea, eb) for eb in embeddings_b]
        for ea in embeddings_a
    ], dtype=np.float32)   # shape (len_a, len_b)

    matched_b = set()
    matches   = []

    # For each row (a), pick the best unmatched column (b)
    for i in range(len(embeddings_a)):
        row = sim_matrix[i].copy()
        # Mask already-matched columns
        for j in matched_b:
            row[j] = -1.0

        best_j   = int(np.argmax(row))
        best_sim = float(row[best_j])

        if best_sim >= threshold:
            matches.append({
                "idx_a":       i,
                "idx_b":       best_j,
                "similarity":  round(best_sim, 4),
            })
            matched_b.add(best_j)

    return matches


# ══════════════════════════════════════════════════════════════════
# IDENTITY GALLERY (persistent across frames / clips)
# ══════════════════════════════════════════════════════════════════

class IdentityGallery:
    """
    Maintains a rolling gallery of known person identities.
    Each identity is represented by an exponential moving average
    of its embeddings so the gallery adapts to appearance changes
    (lighting, clothing angle, etc.).

    Usage
    ─────
    gallery = IdentityGallery()
    pid     = gallery.assign(embedding)   # returns int person_id
    gallery.update(pid, embedding)        # refine appearance model
    """

    def __init__(self, threshold: float = 0.75, ema_alpha: float = 0.3):
        """
        threshold  : cosine similarity required to match an existing identity
        ema_alpha  : weight of new embedding in moving average update
                     (0 = ignore new, 1 = replace fully)
        """
        self.threshold  = threshold
        self.alpha      = ema_alpha
        self._gallery: dict[int, np.ndarray] = {}   # pid → mean embedding
        self._next_id   = 1

    def assign(self, embedding: np.ndarray) -> int:
        """
        Find the best matching existing identity or create a new one.
        Returns the integer person_id.
        """
        best_pid  = None
        best_sim  = self.threshold   # must beat threshold to match

        for pid, gallery_emb in self._gallery.items():
            sim = cosine_similarity(embedding, gallery_emb)
            if sim > best_sim:
                best_sim = sim
                best_pid = pid

        if best_pid is None:
            # New identity
            best_pid = self._next_id
            self._gallery[best_pid] = embedding.copy()
            self._next_id += 1
        else:
            # Update EMA
            self.update(best_pid, embedding)

        return best_pid

    def update(self, pid: int, embedding: np.ndarray):
        """Refine the gallery embedding for an existing identity."""
        if pid not in self._gallery:
            self._gallery[pid] = embedding.copy()
            return
        updated = (1 - self.alpha) * self._gallery[pid] + self.alpha * embedding
        self._gallery[pid] = ReIDModel._l2_normalize(updated)

    def known_ids(self) -> list[int]:
        return list(self._gallery.keys())

    def get_embedding(self, pid: int) -> np.ndarray | None:
        return self._gallery.get(pid)

    def reset(self):
        self._gallery.clear()
        self._next_id = 1


# ══════════════════════════════════════════════════════════════════
# POV SWITCHER
# ══════════════════════════════════════════════════════════════════

class POVSwitcher:
    """
    Manages multiple video sources (POVs / cameras) and decides
    which camera feed to show at each moment based on:
      - A user-requested target person ID
      - Which camera currently has the best view of that person
        (highest similarity + largest bbox area)

    Usage
    ─────
    switcher = POVSwitcher(camera_paths=["cam1.mp4", "cam2.mp4"])
    switcher.set_target(person_id=1)
    result = switcher.best_camera_for_frame(frame_index=42)
    # result → { "camera_index", "camera_path", "similarity", "bbox" }
    """

    def __init__(self,
                 camera_paths: list[str],
                 similarity_threshold: float = 0.70):
        self.camera_paths  = camera_paths
        self.threshold     = similarity_threshold
        self.gallery       = IdentityGallery(threshold=similarity_threshold)
        self.reid_model    = get_reid_model()
        self._target_pid: int | None = None

        # Cache: camera_index → {frame_index → list of (pid, bbox, sim)}
        self._frame_cache: dict[int, dict] = defaultdict(dict)

    def set_target(self, person_id: int):
        """Set which person identity to follow across cameras."""
        self._target_pid = person_id

    # ── Offline: pre-process all cameras ──────────────────────────
    def build_index(self,
                     sample_rate: int = 15,
                     progress_cb=None) -> dict:
        """
        Step 1: Run detection + Re-ID on every camera, building a
        frame-level index of { person_id → bbox } for each camera.

        Call this once before using best_camera_for_frame().

        Returns
        ───────
        {
          camera_index: {
            frame_index: [
              { "person_id", "bbox", "embedding" }
            ]
          }
        }
        """
        from subject_tracker import get_det_model, get_pose_model, detect_frame

        det_model  = get_det_model()
        pose_model = get_pose_model()
        index      = {}

        for cam_idx, cam_path in enumerate(self.camera_paths):
            print(f"[POVSwitcher] Indexing camera {cam_idx}: {cam_path}")
            cap          = cv2.VideoCapture(cam_path)
            fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cam_index    = {}
            frame_idx    = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    detections = detect_frame(frame, det_model, pose_model)
                    crops_info = extract_person_crops(frame, detections)

                    if crops_info:
                        crops      = [c["crop"] for c in crops_info]
                        embeddings = self.reid_model.extract_embeddings_batch(crops)

                        frame_entries = []
                        for ci, emb in zip(crops_info, embeddings):
                            pid = self.gallery.assign(emb)
                            frame_entries.append({
                                "person_id": pid,
                                "bbox":      ci["bbox"],
                                "embedding": emb,
                            })
                        cam_index[frame_idx] = frame_entries

                    if progress_cb and frame_idx % (sample_rate * 10) == 0:
                        overall = (cam_idx / len(self.camera_paths)
                                   + (frame_idx / max(total, 1))
                                   / len(self.camera_paths))
                        progress_cb(int(overall * 100))

                frame_idx += 1

            cap.release()
            index[cam_idx] = cam_index
            self._frame_cache[cam_idx] = cam_index

        if progress_cb:
            progress_cb(100)

        return index

    # ── Online: pick best camera for a given frame ─────────────────
    def best_camera_for_frame(self, frame_index: int) -> dict | None:
        """
        Given a frame index, return which camera has the best view
        of the current target person.

        Looks at the nearest cached sample frame (±sample_rate/2).

        Returns
        ───────
        {
          "camera_index" : int,
          "camera_path"  : str,
          "person_id"    : int,
          "bbox"         : dict,
          "similarity"   : float,
        }
        or None if target not found in any camera.
        """
        if self._target_pid is None:
            return None

        target_emb = self.gallery.get_embedding(self._target_pid)
        if target_emb is None:
            return None

        best_result = None
        best_score  = -1.0

        for cam_idx, cam_index in self._frame_cache.items():
            if not cam_index:
                continue

            # Find nearest cached frame
            cached_frames = sorted(cam_index.keys())
            nearest = min(cached_frames, key=lambda f: abs(f - frame_index))
            entries = cam_index.get(nearest, [])

            for entry in entries:
                if entry["person_id"] != self._target_pid:
                    continue

                sim  = cosine_similarity(entry["embedding"], target_emb)
                bbox = entry["bbox"]
                # Score = similarity × relative bbox area (prefer close-up views)
                area_score = (bbox["w"] * bbox["h"]) ** 0.3   # soft area boost
                score = sim * area_score

                if score > best_score:
                    best_score  = score
                    best_result = {
                        "camera_index": cam_idx,
                        "camera_path":  self.camera_paths[cam_idx],
                        "person_id":    self._target_pid,
                        "bbox":         bbox,
                        "similarity":   round(sim, 4),
                    }

        return best_result

    # ── Convenience: generate a full POV timeline ──────────────────
    def generate_pov_timeline(self,
                               target_person_id: int,
                               sample_rate: int = 15) -> list[dict]:
        """
        For every sampled frame across all cameras, determine which
        camera gives the best view of target_person_id and return
        a switchable timeline.

        Returns
        ───────
        List of:
        {
          "frame_index"   : int,
          "camera_index"  : int,
          "camera_path"   : str,
          "bbox"          : dict,
          "similarity"    : float,
        }
        Sorted by frame_index.
        """
        self.set_target(target_person_id)

        # Collect all frame indices across all cameras
        all_frames = set()
        for cam_index in self._frame_cache.values():
            all_frames.update(cam_index.keys())

        timeline = []
        for fi in sorted(all_frames):
            result = self.best_camera_for_frame(fi)
            if result:
                timeline.append({
                    "frame_index":  fi,
                    **result,
                })

        return timeline


# ══════════════════════════════════════════════════════════════════
# CONVENIENCE: single-call cross-camera match
# ══════════════════════════════════════════════════════════════════

def match_persons_across_cameras(
    frame_a: np.ndarray,
    detections_a: list,
    frame_b: np.ndarray,
    detections_b: list,
    threshold: float = 0.75,
) -> list[dict]:
    """
    One-shot: given two frames + their detections, return which
    persons in frame_a match which persons in frame_b.

    Parameters
    ──────────
    frame_a / frame_b       : BGR frames from two cameras
    detections_a / b        : output of subject_tracker.detect_frame()
    threshold               : cosine similarity threshold

    Returns
    ───────
    List of:
    {
      "cam_a_detection_id" : int,
      "cam_b_detection_id" : int,
      "similarity"         : float,
      "cam_a_bbox"         : dict,
      "cam_b_bbox"         : dict,
    }
    """
    model    = get_reid_model()
    crops_a  = extract_person_crops(frame_a, detections_a)
    crops_b  = extract_person_crops(frame_b, detections_b)

    if not crops_a or not crops_b:
        return []

    embs_a = model.extract_embeddings_batch([c["crop"] for c in crops_a])
    embs_b = model.extract_embeddings_batch([c["crop"] for c in crops_b])

    raw_matches = match_identities(embs_a, embs_b, threshold=threshold)

    results = []
    for m in raw_matches:
        results.append({
            "cam_a_detection_id": crops_a[m["idx_a"]]["detection_id"],
            "cam_b_detection_id": crops_b[m["idx_b"]]["detection_id"],
            "similarity":         m["similarity"],
            "cam_a_bbox":         crops_a[m["idx_a"]]["bbox"],
            "cam_b_bbox":         crops_b[m["idx_b"]]["bbox"],
        })

    return results