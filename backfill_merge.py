"""
Run this once to merge all existing tracking JSONs into their
corresponding result JSONs in the uploads folder.

Usage:
  cd D:\Coding\Videos\effect_analyzer
  python backfill_merge.py
"""
import os, json, sys
from pathlib import Path

# Add project root to path so we can import cinema_analyzer
sys.path.insert(0, str(Path(__file__).parent))
from cinema_analyzer import analyze_video_cinema

UPLOADS = Path("uploads")

def merge(result_path: Path, tracking_path: Path, video_path: Path):
    print(f"\n{'='*60}")
    print(f"  Video:    {video_path.name}")
    print(f"  Result:   {result_path.name}")
    print(f"  Tracking: {tracking_path.name}")

    with open(result_path) as f:
        metadata = json.load(f)

    with open(tracking_path) as f:
        tracking_data = json.load(f)

    # Build detections_by_frame lookup
    detections_by_frame = {}
    for shot in tracking_data.get("shots", []):
        for frame in shot.get("frames", []):
            detections_by_frame[frame["frame"]] = frame["detections"]

    print(f"  Frames with detections: {len(detections_by_frame)}")

    if not video_path.exists():
        print(f"  [SKIP] Video file not found: {video_path}")
        return False

    print(f"  Re-running cinema analysis with tracking data...")
    cinema = analyze_video_cinema(
        str(video_path),
        sample_rate=15,
        detections_by_frame=detections_by_frame,
    )

    if cinema:
        metadata.setdefault("advanced_metadata", {})["cinematography"] = {
            "clip_summary": cinema["summary"],
            "segments":     cinema["segments"],
        }
        print(f"  Cinema segments: {len(cinema['segments'])}")
    else:
        print(f"  [WARN] Cinema analysis returned None")

    metadata.setdefault("advanced_metadata", {})["subject_tracking"] = \
        tracking_data.get("clip_summary", {})

    with open(result_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Merged successfully")
    return True


def main():
    print(f"Scanning uploads folder: {UPLOADS.resolve()}")

    # Find all tracking JSONs
    tracking_files = list(UPLOADS.glob("*_tracking.json"))
    print(f"Found {len(tracking_files)} tracking file(s)")

    merged   = 0
    skipped  = 0
    no_video = 0

    for tracking_path in sorted(tracking_files):
        # Derive the original video filename
        # tracking filename format: <videoname>_tracking.json
        # e.g. v_PlayingGuitar_g03_c04.avi_tracking.json
        stem = tracking_path.name.replace("_tracking.json", "")
        # stem is now e.g. v_PlayingGuitar_g03_c04.avi

        video_path  = UPLOADS / stem
        result_path = UPLOADS / (stem + "_result.json")

        if not result_path.exists():
            print(f"\n  [SKIP] No result JSON for {stem} — run Analyze first")
            skipped += 1
            continue

        if not video_path.exists():
            print(f"\n  [SKIP] Video missing: {stem}")
            no_video += 1
            continue

        ok = merge(result_path, tracking_path, video_path)
        if ok:
            merged += 1

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Merged:         {merged}")
    print(f"  Skipped (no result JSON): {skipped}")
    print(f"  Skipped (no video):       {no_video}")
    print(f"\nRefresh the app and re-open any analyzed video to see overlays.")


if __name__ == "__main__":
    main()