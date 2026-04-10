from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os, json, threading, time
from pathlib import Path
from model    import clf, EFFECTS
from analyzer import extract_features, build_metadata

app          = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Global training state
train_state = {"status": "idle", "progress": 0, "result": None}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    def run():
        train_state["status"]   = "training"
        train_state["progress"] = 0
        def cb(p):
            train_state["progress"] = p
        result = clf.train(progress_cb=cb)
        train_state["status"]   = "done"
        train_state["progress"] = 100
        train_state["result"]   = result
    threading.Thread(target=run).start()
    return jsonify({"started": True})

@app.route("/train/status")
def train_status():
    return jsonify(train_state)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f    = request.files["video"]
    ext  = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    fname = secure_filename(f.filename)
    path  = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    f.save(path)

    if not clf.trained:
        return jsonify({"error": "Model not trained yet. Click Train first."}), 400

    features = extract_features(path)
    if features is None:
        return jsonify({"error": "Could not process video — too short or corrupted."}), 400

    predictions = clf.predict(features)
    metadata    = build_metadata(path, predictions, features)

    result_path = os.path.join(app.config["UPLOAD_FOLDER"], fname + "_result.json")
    with open(result_path, "w") as out:
        json.dump(metadata, out, indent=2)

    return jsonify({"metadata": metadata, "result_file": fname + "_result.json"})

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(filename))
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    print("Starting EffectScan at http://localhost:5000")
    app.run(debug=True, port=5000)