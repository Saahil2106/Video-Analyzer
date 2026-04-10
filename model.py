import json
import numpy as np
from pathlib import Path
from collections import defaultdict

METADATA_PATH = "D:/Coding/Videos/datasets/processed/clip_metadata_v2.json"
EFFECTS = [
    "speed_ramp", "freeze_moment", "rapid_motion",
    "rhythmic_motion", "speed_variation", "spin_rotation", "smooth_motion"
]
FEATURES = [
    "mean_flow", "max_flow", "std_flow", "flow_variance",
    "frame_diff_mean", "frame_diff_std", "freq_ratio",
    "freeze_ratio", "accel_score", "periodicity_score",
    "rotation_score", "flow_gradient"
]

class EffectClassifier:
    def __init__(self):
        self.means   = {}
        self.stds    = {}
        self.trained = False

    def load_data(self):
        with open(METADATA_PATH) as f:
            data = json.load(f)
        grouped = defaultdict(list)
        for clip in data:
            vec = [clip.get(feat, 0.0) for feat in FEATURES]
            for effect in clip["effects"]:
                grouped[effect].append(vec)
        return grouped

    def train(self, progress_cb=None):
        grouped = self.load_data()
        for i, effect in enumerate(EFFECTS):
            vecs = np.array(grouped[effect])
            self.means[effect] = vecs.mean(axis=0)
            self.stds[effect]  = vecs.std(axis=0) + 1e-8
            if progress_cb:
                progress_cb(int((i + 1) / len(EFFECTS) * 100))
        self.trained = True
        return self.evaluate(grouped)

    def evaluate(self, grouped):
        correct, total = 0, 0
        f1_scores = {}
        for effect in EFFECTS:
            vecs = grouped.get(effect, [])
            if not vecs:
                continue
            tp = fp = fn = 0
            for vec in vecs:
                pred = self.predict_vec(np.array(vec))
                if effect in pred:
                    tp += 1
                else:
                    fn += 1
            for other_effect, other_vecs in grouped.items():
                if other_effect == effect:
                    continue
                for vec in other_vecs[:50]:
                    pred = self.predict_vec(np.array(vec))
                    if effect in pred:
                        fp += 1
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores[effect] = round(f1, 3)
            correct += tp
            total   += tp + fn
        accuracy = round(correct / total * 100, 1) if total else 0
        return {"accuracy": accuracy, "f1_scores": f1_scores}

    def predict_vec(self, vec):
        scores = {}
        for effect in EFFECTS:
            if effect not in self.means:
                continue
            z     = np.abs((vec - self.means[effect]) / self.stds[effect])
            score = float(np.exp(-z.mean()))
            scores[effect] = score
        threshold = 0.25
        detected  = {e: s for e, s in scores.items() if s >= threshold}
        if not detected:
            detected = {max(scores, key=scores.get): max(scores.values())}
        return detected

    def predict(self, features: dict):
        vec    = np.array([features.get(f, 0.0) for f in FEATURES])
        scores = self.predict_vec(vec)
        total  = sum(scores.values()) + 1e-8
        return {e: round(s / total, 3) for e, s in sorted(
            scores.items(), key=lambda x: -x[1]
        )}

clf = EffectClassifier()