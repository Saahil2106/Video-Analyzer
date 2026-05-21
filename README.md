# Video Analyzer

Professional local installation guide for the VIDEO-ANALYZER project.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A local Flask-based video analysis platform for uploading videos, analyzing visual effects, tracking subjects, and extracting per-frame features.

## Features

- Video upload and playback.
- Automated analysis and metadata generation.
- Subject tracking with YOLO-based detection and pose estimation.
- Feature extraction for motion, color, brightness, texture, and scene structure.
- JSON result generation and HTML report viewing.

## Project structure

```text
VIDEO-ANALYZER/
├── app.py
├── analyzer.py
├── backfill_merge.py
├── cinema_analyzer.py
├── feature_extractor.py
├── json_utils.py
├── model.py
├── pipeline.py
├── subject_tracker.py
├── tracker/
│   ├── __init__.py
│   ├── kalman_tracker.py
│   ├── mct.py
│   └── reid_tracker.py
├── templates/
│   ├── index.html
│   └── report.html
├── yolo11m-pose.pt
├── yolo11m.pt
├── yolov8n-pose.pt
├── yolov8n.pt
├── yolov8x-worldv2.pt
└── UCF101.rar
```

## Requirements

- Python 3.10 or newer.
- pip.
- ffmpeg.

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd VIDEO-ANALYZER
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install flask werkzeug opencv-python ultralytics numpy scipy
```

If you need the extra OpenCV trackers used by the tracking package:

```bash
pip uninstall -y opencv-python
pip install opencv-contrib-python
```

### 4. Install ffmpeg

Windows:

```powershell
winget install ffmpeg
```

macOS:

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

## Run locally

```bash
python app.py
```

Open the app in your browser at:

```text
http://localhost:5000
```

## Usage

1. Upload a supported video file.
2. Run analysis, tracking, or feature extraction.
3. View the generated report in the browser.
4. Download the resulting JSON files from the UI.

## Supported formats

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`

## Model files

The project expects these model weights in the repository root:

- `yolo11m-pose.pt`
- `yolo11m.pt`
- `yolov8n-pose.pt`
- `yolov8n.pt`
- `yolov8x-worldv2.pt`

If they are present, the app can run locally without redownloading them.

## Output files

Generated files are saved to the `uploads/` folder, including:

- `<video_name>_result.json`
- `<video_name>_tracking.json`
- `<video_name>_features.json`

## Deployment notes

- Keep the `templates/` folder in the project root so Flask can render the UI and report pages.
- Keep the `tracker/` package intact because it contains tracking utilities used by the application.
- Run the app from the project root so relative paths resolve correctly.

## Troubleshooting

### Template not found

Make sure `index.html` and `report.html` are inside `templates/`.

### ffmpeg not found

Install ffmpeg and confirm it is available in your system PATH.

### OpenCV tracker errors

Install `opencv-contrib-python` if a tracker backend is missing.

### Model loading issues

Verify the `.pt` files exist and that Ultralytics and PyTorch are installed correctly.

## License

Add your preferred license here before publishing on GitHub.
