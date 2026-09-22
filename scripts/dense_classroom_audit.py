"""Detector/tracker-only dense classroom audit.

This script never loads the ResNet/temporal model and writes only diagnostic
artifacts under storage/dense_classroom_audit.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.detector import TiledYOLOPersonDetector, YOLOPersonDetector
from app.ml.appearance import AnonymousAppearanceEmbedder
from app.ml.tracker import ByteTracker, DenseByteTracker


MANUAL_REFERENCE_ESTIMATE = 49
SAMPLE_FPS = 2.0
OUTPUT_DIR = Path("storage/dense_classroom_audit")


def read_sampled_frames(video_path: str) -> Tuple[dict, List[Tuple[int, float, np.ndarray]]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    metadata = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "sample_fps": SAMPLE_FPS,
    }
    interval = max(1, int(round(fps / SAMPLE_FPS)))
    frames = []
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if frame_number % interval == 0:
            frames.append((frame_number, frame_number / fps, frame.copy()))
        frame_number += 1
    capture.release()
    if not frames:
        raise ValueError("Video contains no decodable sampled frames")
    metadata["sampled_frames"] = len(frames)
    metadata["sample_interval_native_frames"] = interval
    return metadata, frames


def draw_detections(frame, detections, frame_number, title):
    image = frame.copy()
    cv2.putText(image, f"{title} | frame {frame_number}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(image, f"person {detection['confidence']:.2f}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    return image


def draw_tracks(frame, active_tracks, frame_number, title):
    image = frame.copy()
    cv2.putText(image, f"{title} | frame {frame_number} | active {len(active_tracks)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
    for track in active_tracks:
        x1, y1, x2, y2 = [int(value) for value in track.bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Student {track.track_id:02d} {track.score:.2f}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 2)
    return image


def detections_from_result(result, frame_number, timestamp):
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        if int(box.cls[0].cpu().numpy()) != 0:
            continue
        detections.append({
            "bbox": box.xyxy[0].cpu().numpy().tolist(),
            "confidence": float(box.conf[0].cpu().numpy()),
            "class_name": "person",
            "frame_idx": frame_number,
            "timestamp": timestamp,
        })
    return detections


def run_full_detector(model, frames, confidence, image_size, iou):
    outputs = []
    for frame_number, timestamp, frame in frames:
        result = model.predict(source=frame, classes=[0], conf=confidence, imgsz=image_size, iou=iou, verbose=False)[0]
        outputs.append(detections_from_result(result, frame_number, timestamp))
    return outputs


def run_tiled_detector(detector, frames):
    merged = []
    raw = []
    for frame_number, timestamp, frame in frames:
        merged.append(detector.detect_persons(frame, frame_number, timestamp))
        raw.append(detector.last_raw_detections)
    return raw, merged


def collect_metrics(name, raw, merged, active_by_frame, tracker, metadata, configuration):
    all_tracks = tracker.get_all_observed_tracks() if hasattr(tracker, "get_all_observed_tracks") else tracker.get_all_session_tracks()
    confirmed_tracks = tracker.get_all_session_tracks()
    durations = [len(track.history) for track in all_tracks]
    interval = metadata["sample_interval_native_frames"]
    reappearances = sum(
        1 for track in all_tracks
        if any((later["frame"] - earlier["frame"]) > interval for earlier, later in zip(track.history, track.history[1:]))
    )
    short_tracks = sum(1 for duration in durations if duration < 3)
    stable_tracks = sum(1 for duration in durations if duration >= max(5, metadata["sampled_frames"] // 4))
    max_confirmed = max((len(active) for active in active_by_frame), default=0)
    return {
        "configuration": name,
        "detector_configuration": configuration,
        "raw_detections_mean": statistics.mean(len(items) for items in raw),
        "raw_detections_min": min(len(items) for items in raw),
        "raw_detections_max": max(len(items) for items in raw),
        "merged_detections_mean": statistics.mean(len(items) for items in merged),
        "merged_detections_max": max(len(items) for items in merged),
        "confirmed_tracks_max_simultaneous": max_confirmed,
        "unique_session_tracks": len(confirmed_tracks),
        "unique_observed_tracks_including_unconfirmed": len(all_tracks),
        "average_track_duration_sampled_frames": statistics.mean(durations) if durations else 0,
        "median_track_duration_sampled_frames": statistics.median(durations) if durations else 0,
        "track_fragmentation": len(all_tracks) - stable_tracks,
        "track_reappearance_count": reappearances,
        "short_false_track_count_heuristic": short_tracks,
        "long_stable_track_count_heuristic": stable_tracks,
        "manual_reference_estimate": MANUAL_REFERENCE_ESTIMATE,
        "peak_confirmed_track_coverage_against_manual_reference": max_confirmed / MANUAL_REFERENCE_ESTIMATE,
    }


def write_video(path: Path, frames: List[np.ndarray], width: int, height: int):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), SAMPLE_FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create diagnostic video: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=r"C:\Users\aruna\Downloads\classroom.mp4")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata, frames = read_sampled_frames(args.video)
    print(json.dumps({"video": args.video, **metadata}))

    model_n = YOLO("yolov8n.pt")
    model_s = YOLO("yolov8s.pt")
    configurations = [
        ("A_n_full", "full", model_n, 0.10, 640, 0.70, False),
        ("B_s_full_1280", "full", model_s, 0.15, 1280, 0.55, False),
    ]
    reports = []
    rendered_detection = []
    rendered_tracking = []
    rendered_final = []
    appearance_embedder = AnonymousAppearanceEmbedder()

    for name, mode, model, confidence, image_size, iou, dense in configurations:
        raw = run_full_detector(model, frames, confidence, image_size, iou)
        merged = raw
        tracker = ByteTracker(track_thresh=0.30, match_thresh=0.80, match_thresh_second=0.40, max_time_lost_frames=90)
        active = [tracker.update(items, frame_number, timestamp) for items, (frame_number, timestamp, _) in zip(merged, frames)]
        reports.append(collect_metrics(name, raw, merged, active, tracker, metadata, {"model": name.split("_")[1], "mode": mode, "confidence": confidence, "imgsz": image_size, "iou": iou, "tracker": "baseline"}))
        if name == "B_s_full_1280":
            rendered_detection = [draw_detections(frame, items, frame_number, name) for items, (frame_number, _, frame) in zip(raw, frames)]
            rendered_tracking = [draw_tracks(frame, items, frame_number, name) for items, (frame_number, _, frame) in zip(active, frames)]

    tiled = TiledYOLOPersonDetector(
        weights_path="yolov8s.pt",
        confidence_threshold=0.15,
        image_size=1280,
        iou_threshold=0.55,
        tile_size=640,
        tile_overlap=0.25,
        merge_iou=0.55,
    )
    raw_tiled, merged_tiled = run_tiled_detector(tiled, frames)
    for name, merged, tracker_class, tracker_args in [
        ("C_s_tiled", raw_tiled, ByteTracker, {"track_thresh": 0.30, "match_thresh": 0.80, "match_thresh_second": 0.40, "max_time_lost_frames": 90}),
        ("D_s_tiled_suppressed", merged_tiled, ByteTracker, {"track_thresh": 0.30, "match_thresh": 0.80, "match_thresh_second": 0.40, "max_time_lost_frames": 90}),
        ("E_s_tiled_suppressed_dense", merged_tiled, DenseByteTracker, {"track_thresh": 0.30, "match_thresh": 0.80, "match_thresh_second": 0.40, "max_time_lost_frames": 12, "confirmation_hits": 2, "center_distance_gate": 2.5}),
    ]:
        tracker = tracker_class(**tracker_args)
        active = []
        for items, (frame_number, timestamp, frame) in zip(merged, frames):
            if isinstance(tracker, DenseByteTracker):
                for item in items:
                    item["appearance"] = appearance_embedder(frame, item["bbox"])
            active.append(tracker.update(items, frame_number, timestamp))
        reports.append(collect_metrics(name, raw_tiled, merged, active, tracker, metadata, {"model": "yolov8s.pt", "mode": "tiled", "confidence": 0.15, "imgsz": 1280, "iou": 0.55, "tile_size": 640, "tile_overlap": 0.25, "merge_iou": 0.55, "tracker": tracker_class.__name__, **tracker_args}))
        if name == "E_s_tiled_suppressed_dense":
            rendered_final = [draw_tracks(frame, items, frame_number, name) for items, (frame_number, _, frame) in zip(active, frames)]

    report = {"video": args.video, "video_metadata": metadata, "manual_reference_note": "49 is a manual reference estimate, not automated ground truth.", "reports": reports}
    (OUTPUT_DIR / "dense_classroom_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_video(OUTPUT_DIR / "detection_only_yolov8s_full_1280.mp4", rendered_detection, metadata["width"], metadata["height"])
    write_video(OUTPUT_DIR / "tracking_only_yolov8s_full_1280.mp4", rendered_tracking, metadata["width"], metadata["height"])
    write_video(OUTPUT_DIR / "final_dense_tracking_diagnostic.mp4", rendered_final, metadata["width"], metadata["height"])
    representative_indices = {
        "beginning": 0,
        "middle": len(rendered_final) // 2,
        "end": len(rendered_final) - 1,
    }
    for label, index in representative_indices.items():
        cv2.imwrite(str(OUTPUT_DIR / f"final_dense_tracking_{label}.jpg"), rendered_final[index])
    print(json.dumps(report, indent=2))
    print(f"outputs={OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
