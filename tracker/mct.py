"""
Multi-Camera Tracker (MCT).
Re-identifies targets seen in Camera A inside Camera B's detections
using an appearance embedding (histogram of HOG + colour) and
a lightweight cosine-similarity gallery.

Usage:
    mct = MultiCameraTracker(cameras=["cam_left", "cam_right"])
    # Per frame, for each camera:
    mct.update("cam_left",  frame_idx, detections_left)
    mct.update("cam_right", frame_idx, detections_right)
    # Query global ID for a local detection:
    global_id = mct.get_global_id("cam_right", local_track_id)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib


# ─────────────────────────────────────────────────────────────
# Appearance embedding (pure-numpy, no deep-learning required)
# ─────────────────────────────────────────────────────────────

def _colour_histogram(patch: np.ndarray, bins: int = 16) -> np.ndarray:
    """Fast HSV-like histogram from an RGB/BGR patch array."""
    if patch is None or patch.size == 0:
        return np.zeros(bins * 3, dtype=np.float32)
    patch = patch.astype(np.float32) / 255.0
    hists = []
    for ch in range(min(3, patch.shape[-1])):
        h, _ = np.histogram(patch[..., ch].flatten(), bins=bins, range=(0, 1))
        hists.append(h.astype(np.float32))
    feat = np.concatenate(hists)
    n = np.linalg.norm(feat)
    return feat / n if n > 0 else feat


def _hog_simple(patch: np.ndarray, cells: int = 4) -> np.ndarray:
    """Minimal HOG gradient orientation histogram (no cv2 dep)."""
    if patch is None or patch.size < 4:
        return np.zeros(cells * cells * 9, dtype=np.float32)
    gray = patch.mean(axis=-1) if patch.ndim == 3 else patch
    gray = gray.astype(np.float32)
    gy = np.gradient(gray, axis=0)
    gx = np.gradient(gray, axis=1)
    mag   = np.sqrt(gx**2 + gy**2)
    angle = (np.arctan2(gy, gx) * 180 / np.pi) % 180
    h, w  = gray.shape
    ch, cw = max(1, h // cells), max(1, w // cells)
    feats  = []
    for i in range(cells):
        for j in range(cells):
            cell_mag   = mag  [i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            cell_angle = angle[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            hist, _    = np.histogram(cell_angle.flatten(), bins=9,
                                      range=(0, 180),
                                      weights=cell_mag.flatten())
            feats.append(hist.astype(np.float32))
    feat = np.concatenate(feats)
    n = np.linalg.norm(feat)
    return feat / n if n > 0 else feat


def extract_embedding(frame: Optional[np.ndarray],
                      bbox: dict,
                      patch_size: Tuple[int,int] = (64, 32)) -> np.ndarray:
    """
    Extract appearance embedding for a detection.
    Falls back to a deterministic hash when no frame pixels are available.
    """
    if frame is not None:
        x  = max(0, int(bbox.get("x", 0)))
        y  = max(0, int(bbox.get("y", 0)))
        w  = max(1, int(bbox.get("w", 1)))
        h  = max(1, int(bbox.get("h", 1)))
        patch = frame[y:y+h, x:x+w]
        if patch.size > 0:
            try:
                import cv2
                patch = cv2.resize(patch, patch_size)
            except Exception:
                # fallback resize
                ph, pw = patch_size[1], patch_size[0]
                patch  = patch[::max(1,patch.shape[0]//ph),
                               ::max(1,patch.shape[1]//pw)]
            colour = _colour_histogram(patch)
            hog    = _hog_simple(patch)
            emb    = np.concatenate([colour, hog])
            n      = np.linalg.norm(emb)
            return emb / n if n > 0 else emb

    # Deterministic fallback using bbox geometry
    key = f"{bbox.get('x',0):.1f},{bbox.get('y',0):.1f},{bbox.get('w',0):.1f},{bbox.get('h',0):.1f}"
    digest = hashlib.md5(key.encode()).digest()
    arr = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    arr = np.tile(arr, 10)[:144]
    n   = np.linalg.norm(arr)
    return arr / n if n > 0 else arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────
# Gallery — per-global-ID rolling embedding average
# ─────────────────────────────────────────────────────────────

@dataclass
class GalleryEntry:
    global_id:   int
    label:       str
    embedding:   np.ndarray          # EMA-updated representative vector
    appearances: int = 1             # number of times observed
    cameras:     Dict[str,int] = field(default_factory=dict)  # cam_id → local_track_id
    last_frame:  int = 0
    confirmed:   bool = False

    def update_embedding(self, new_emb: np.ndarray, alpha: float = 0.3):
        """Exponential moving average update."""
        self.embedding = alpha * new_emb + (1 - alpha) * self.embedding
        n = np.linalg.norm(self.embedding)
        if n > 0:
            self.embedding /= n
        self.appearances += 1
        self.confirmed    = self.appearances >= 3


class AppearanceGallery:
    """Thread-safe (single-threaded use) gallery of global identities."""

    REID_THRESHOLD  = 0.65    # cosine similarity threshold for re-ID match
    MAX_GALLERY_AGE = 500     # frames before a gallery entry is pruned

    def __init__(self):
        self.entries: Dict[int, GalleryEntry] = {}
        self._next_id = 1

    def match_or_register(self, embedding: np.ndarray,
                          label: str,
                          camera_id: str,
                          local_track_id: int,
                          frame_idx: int) -> int:
        """
        Compare embedding against gallery. If similarity ≥ threshold,
        update the existing entry and return its global_id.
        Otherwise register a new global identity.
        Returns global_id (int).
        """
        best_id    = None
        best_score = self.REID_THRESHOLD - 1e-6

        for gid, entry in self.entries.items():
            if entry.label != label:
                continue
            # Don't match if the same camera already owns this global track
            # on this frame (prevents same-camera duplicates)
            if camera_id in entry.cameras and \
               entry.cameras[camera_id] == local_track_id and \
               entry.last_frame == frame_idx:
                continue
            score = cosine_similarity(embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_id    = gid

        if best_id is not None:
            e = self.entries[best_id]
            e.update_embedding(embedding)
            e.cameras[camera_id] = local_track_id
            e.last_frame         = frame_idx
            return best_id
        else:
            new_entry = GalleryEntry(
                global_id   = self._next_id,
                label       = label,
                embedding   = embedding.copy(),
                cameras     = {camera_id: local_track_id},
                last_frame  = frame_idx,
            )
            self.entries[self._next_id] = new_entry
            self._next_id += 1
            return new_entry.global_id

    def prune(self, current_frame: int):
        stale = [gid for gid, e in self.entries.items()
                 if current_frame - e.last_frame > self.MAX_GALLERY_AGE]
        for gid in stale:
            del self.entries[gid]


# ─────────────────────────────────────────────────────────────
# MultiCameraTracker — top-level orchestrator
# ─────────────────────────────────────────────────────────────

class MultiCameraTracker:
    """
    Coordinates Kalman trackers for each camera and the shared
    appearance gallery for cross-camera re-identification.

    Example integration in your Flask /track route:

        mct = MultiCameraTracker(cameras=["cam0"])
        for frame_idx, dets in enumerate(yolo_detections_per_frame):
            enriched = mct.update("cam0", frame_idx, dets, frame_pixels=frame_np)
            # enriched dets now have: track_id, global_id, predicted_bbox,
            #   velocity, confirmed, reid_score, seen_in_cameras
    """

    def __init__(self, cameras: List[str] = None):
        from tracker.kalman_tracker import KalmanMultiTracker
        self.cameras   = cameras or ["cam0"]
        self.kf_trackers: Dict[str, KalmanMultiTracker] = {
            cam: KalmanMultiTracker(camera_id=cam) for cam in self.cameras
        }
        self.gallery  = AppearanceGallery()
        self._local_to_global: Dict[Tuple[str,int], int] = {}
        # (camera_id, local_track_id) → global_id

    def add_camera(self, camera_id: str):
        if camera_id not in self.kf_trackers:
            from tracker.kalman_tracker import KalmanMultiTracker
            self.kf_trackers[camera_id] = KalmanMultiTracker(camera_id=camera_id)
            self.cameras.append(camera_id)

    def update(self,
               camera_id:    str,
               frame_idx:    int,
               detections:   List[dict],
               frame_pixels: Optional[np.ndarray] = None) -> List[dict]:
        """
        Run Kalman tracking + Re-ID for one camera's frame.

        Args:
            camera_id:    camera identifier string
            frame_idx:    absolute frame index (used for gallery pruning)
            detections:   raw YOLO detection dicts
                          [{bbox:{x,y,w,h}, label, confidence, pose?}, ...]
            frame_pixels: optional numpy HxWxC array for appearance embedding

        Returns:
            Enriched detection list with extra keys:
              track_id       – per-camera Kalman track id
              global_id      – cross-camera re-id global identity
              predicted_bbox – Kalman prediction {x,y,w,h,cx,cy}
              velocity       – {vx, vy} in pixels/frame
              confirmed      – bool (≥3 hits)
              track_age      – frames since track creation
              reid_score     – cosine similarity that triggered re-id
              seen_in_cameras– list of camera ids this global_id was seen in
        """
        if camera_id not in self.kf_trackers:
            self.add_camera(camera_id)

        # Step 1: Kalman filter update
        kf_dets = self.kf_trackers[camera_id].update_frame(detections)

        # Step 2: Appearance-based re-ID for each detection
        enriched = []
        for det in kf_dets:
            local_tid = det.get("track_id")
            if local_tid is None:
                det["global_id"]       = None
                det["reid_score"]      = 0.0
                det["seen_in_cameras"] = [camera_id]
                enriched.append(det)
                continue

            emb = extract_embedding(frame_pixels, det["bbox"])
            gid = self.gallery.match_or_register(
                embedding      = emb,
                label          = det.get("label", "object"),
                camera_id      = camera_id,
                local_track_id = local_tid,
                frame_idx      = frame_idx,
            )
            self._local_to_global[(camera_id, local_tid)] = gid

            entry             = self.gallery.entries[gid]
            det["global_id"]  = gid
            det["reid_score"] = float(cosine_similarity(emb, entry.embedding))
            det["seen_in_cameras"] = list(entry.cameras.keys())
            enriched.append(det)

        # Step 3: Prune stale gallery entries periodically
        if frame_idx % 100 == 0:
            self.gallery.prune(frame_idx)

        return enriched

    def get_global_id(self, camera_id: str, local_track_id: int) -> Optional[int]:
        return self._local_to_global.get((camera_id, local_track_id))

    def get_cross_camera_matches(self) -> List[dict]:
        """
        Return list of global IDs seen in more than one camera
        (the actual cross-camera re-ID results).
        """
        matches = []
        for gid, entry in self.gallery.entries.items():
            if len(entry.cameras) > 1:
                matches.append({
                    "global_id":  gid,
                    "label":      entry.label,
                    "cameras":    dict(entry.cameras),
                    "appearances": entry.appearances,
                    "confirmed":  entry.confirmed,
                })
        return matches

    def summary(self) -> dict:
        return {
            "total_global_ids":    len(self.gallery.entries),
            "cross_camera_matches": len(self.get_cross_camera_matches()),
            "cameras":             self.cameras,
            "gallery_size":        len(self.gallery.entries),
        }