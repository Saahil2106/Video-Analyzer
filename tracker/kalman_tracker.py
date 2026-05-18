"""
Kalman Filter Multi-Object Tracker
Tracks detections across frames using a simple constant-velocity Kalman filter.
No external dependencies beyond numpy.
"""

import numpy as np
from typing import List, Optional


class KalmanTrack:
    """Single object track with Kalman filter state [cx, cy, w, h, vx, vy, vw, vh]."""

    _id_counter = 0

    def __init__(self, bbox: dict, label: str):
        KalmanTrack._id_counter += 1
        self.track_id = KalmanTrack._id_counter
        self.label = label
        self.hits = 1
        self.misses = 0
        self.confirmed = False

        self.CONFIRM_HITS = 2
        self.MAX_MISSES = 8

        x = float(bbox["x"]) + float(bbox["w"]) / 2.0
        y = float(bbox["y"]) + float(bbox["h"]) / 2.0
        w = float(bbox["w"])
        h = float(bbox["h"])

        self.state = np.array([x, y, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.eye(4, 8, dtype=np.float64)

        self.P = np.diag([10.0, 10.0, 10.0, 10.0,
                          100.0, 100.0, 10.0, 10.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0,
                          0.01, 0.01, 0.01, 0.01])
        self.R = np.diag([10.0, 10.0, 10.0, 10.0])

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, bbox: dict):
        z = np.array([
            float(bbox["x"]) + float(bbox["w"]) / 2.0,
            float(bbox["y"]) + float(bbox["h"]) / 2.0,
            float(bbox["w"]),
            float(bbox["h"]),
        ], dtype=np.float64)

        y_innov = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y_innov
        self.P = (np.eye(8, dtype=np.float64) - K @ self.H) @ self.P

        self.hits += 1
        self.misses = 0
        if self.hits >= self.CONFIRM_HITS:
            self.confirmed = True

    def get_bbox(self) -> dict:
        cx, cy, w, h = self.state[:4]
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        return {
            "x": round(float(cx - w / 2.0), 1),
            "y": round(float(cy - h / 2.0), 1),
            "w": round(w, 1),
            "h": round(h, 1),
            "x2": round(float(cx + w / 2.0), 1),
            "y2": round(float(cy + h / 2.0), 1),
            "cx": round(float(cx), 1),
            "cy": round(float(cy), 1),
        }

    def get_velocity(self) -> dict:
        return {
            "vx": round(float(self.state[4]), 2),
            "vy": round(float(self.state[5]), 2),
        }

    def is_alive(self) -> bool:
        return self.misses <= self.MAX_MISSES


def _iou(a: dict, b: dict) -> float:
    ax1, ay1 = float(a["x"]), float(a["y"])
    ax2, ay2 = ax1 + float(a["w"]), ay1 + float(a["h"])
    bx1, by1 = float(b["x"]), float(b["y"])
    bx2, by2 = bx1 + float(b["w"]), by1 + float(b["h"])

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    if inter == 0.0:
        return 0.0

    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class KalmanMultiTracker:
    """
    Manages multiple KalmanTrack instances for one camera.
    Uses greedy IoU matching to associate detections → tracks.
    """

    def __init__(
        self,
        iou_threshold: float = 0.20,
        camera_id: Optional[int] = None,
        **kwargs,
    ):
        self.tracks: List[KalmanTrack] = []
        self.iou_threshold = iou_threshold
        self.camera_id = camera_id
        self.extra_config = dict(kwargs)
        self.frame_index = 0

    def reset(self):
        self.tracks = []
        self.frame_index = 0

    def update(self, detections: List[dict]) -> List[dict]:
        for t in self.tracks:
            t.predict()

        matched_track_indices = set()
        matched_det_ids = set()
        scores = []

        for di, det in enumerate(detections):
            det_label = det.get("label", "unknown")
            det_bbox = det.get("bbox")
            if not det_bbox:
                continue

            for ti, track in enumerate(self.tracks):
                if track.label != det_label:
                    continue
                iou = _iou(det_bbox, track.get_bbox())
                if iou >= self.iou_threshold:
                    scores.append((iou, di, ti))

        scores.sort(reverse=True)

        det_to_track_id = {}
        for iou, di, ti in scores:
            if di in matched_det_ids or ti in matched_track_indices:
                continue
            self.tracks[ti].update(detections[di]["bbox"])
            matched_det_ids.add(di)
            matched_track_indices.add(ti)
            det_to_track_id[di] = self.tracks[ti].track_id

        new_track_ids_for_det = {}
        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                new_track = KalmanTrack(det["bbox"], det.get("label", "unknown"))
                self.tracks.append(new_track)
                new_track_ids_for_det[di] = new_track.track_id

        spawned_ids = set(new_track_ids_for_det.values())
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_indices and track.track_id not in spawned_ids:
                track.misses += 1

        self.tracks = [t for t in self.tracks if t.is_alive()]
        track_by_id = {t.track_id: t for t in self.tracks}

        enriched = []
        seen_track_ids = set()

        for di, det in enumerate(detections):
            d = dict(det)
            assigned_tid = det_to_track_id.get(di)

            if assigned_tid is None:
                assigned_tid = new_track_ids_for_det.get(di)

            if assigned_tid is not None and assigned_tid in track_by_id:
                t = track_by_id[assigned_tid]

                if assigned_tid in seen_track_ids:
                    continue
                seen_track_ids.add(assigned_tid)

                d["track_id"] = t.track_id
                d["confirmed"] = t.confirmed
                d["predicted_bbox"] = t.get_bbox()
                d["velocity"] = t.get_velocity()
                d["track_hits"] = t.hits
                d["track_misses"] = t.misses
            else:
                d["track_id"] = None
                d["confirmed"] = False
                d["predicted_bbox"] = d.get("bbox")
                d["velocity"] = {"vx": 0.0, "vy": 0.0}
                d["track_hits"] = 0
                d["track_misses"] = 0

            enriched.append(d)

        return enriched

    def update_frame(self, detections: List[dict], frame=None, timestamp=None, **kwargs) -> List[dict]:
        """
        Compatibility wrapper for callers expecting update_frame(...).
        Accepts extra args but uses detections as the primary input.
        """
        self.frame_index += 1
        return self.update(detections)

    def get_active_tracks(self) -> List[dict]:
        return [
            {
                "track_id": t.track_id,
                "label": t.label,
                "confirmed": t.confirmed,
                "hits": t.hits,
                "misses": t.misses,
                "bbox": t.get_bbox(),
                "velocity": t.get_velocity(),
            }
            for t in self.tracks
        ]