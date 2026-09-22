"""Manual five-frame benchmark for the appearance-aware dense tracker.

This is an evaluation-only harness. It does not change model weights, tracker
code, checkpoints, datasets, or application defaults.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.appearance import AnonymousAppearanceEmbedder
from app.ml.detector import TiledYOLOPersonDetector
from app.ml.tracker import DenseByteTracker


OUTPUT_DIR = Path("storage/dense_classroom_audit")
BENCHMARK_FRAMES = [0, 86, 171, 256, 342]

# Boxes were manually reviewed on the extracted benchmark frames. They cover
# clearly visible heads/upper bodies, including partial people at image edges.
MANUAL_BOXES = {
    0: [[24, 208, 184, 438], [160, 208, 362, 425], [665, 122, 831, 341], [614, 115, 695, 235], [36, 149, 170, 304], [170, 102, 237, 167], [448, 112, 540, 404], [166, 170, 249, 299], [9, 125, 73, 184], [702, 106, 767, 199], [43, 105, 94, 166], [0, 176, 42, 308], [112, 127, 166, 177], [824, 133, 848, 216], [394, 116, 467, 328], [596, 98, 647, 188], [527, 109, 618, 242], [361, 99, 432, 292], [790, 76, 837, 132], [674, 106, 727, 205], [438, 94, 482, 148], [0, 95, 30, 155], [95, 95, 145, 155], [245, 90, 305, 165], [300, 88, 350, 170], [480, 82, 525, 145], [550, 82, 600, 150], [735, 70, 780, 140]],
    86: [[0, 217, 184, 466], [721, 138, 848, 302], [440, 128, 520, 245], [520, 121, 603, 236], [502, 165, 714, 347], [0, 102, 47, 173], [0, 179, 70, 303], [290, 159, 420, 411], [642, 109, 717, 171], [659, 163, 778, 317], [376, 122, 446, 250], [165, 104, 215, 170], [295, 124, 373, 210], [786, 130, 848, 221], [188, 113, 256, 240], [727, 101, 765, 152], [574, 102, 638, 167], [105, 88, 155, 145], [260, 85, 315, 150], [330, 85, 380, 150], [405, 90, 455, 150], [470, 90, 520, 150], [545, 82, 585, 145], [610, 85, 660, 145], [680, 82, 730, 145], [770, 82, 820, 145]],
    171: [[258, 179, 489, 372], [183, 150, 270, 272], [269, 140, 350, 257], [104, 144, 191, 283], [408, 175, 507, 314], [451, 151, 607, 302], [607, 133, 759, 421], [552, 132, 615, 212], [598, 123, 629, 195], [8, 178, 146, 465], [12, 149, 98, 248], [392, 125, 452, 183], [800, 131, 848, 224], [62, 122, 111, 173], [319, 119, 376, 184], [504, 143, 591, 228], [0, 114, 50, 251], [671, 138, 764, 392], [150, 105, 200, 160], [220, 95, 270, 155], [285, 95, 335, 155], [350, 90, 400, 155], [430, 95, 480, 150], [485, 95, 535, 150], [635, 92, 680, 150], [705, 90, 755, 145], [760, 95, 810, 150]],
    256: [[404, 183, 634, 373], [412, 145, 493, 261], [253, 150, 340, 291], [559, 178, 663, 315], [333, 161, 415, 275], [623, 179, 769, 306], [182, 181, 289, 373], [764, 138, 848, 445], [704, 127, 769, 208], [530, 127, 600, 197], [735, 114, 788, 206], [608, 119, 649, 182], [461, 123, 530, 189], [103, 163, 186, 388], [703, 115, 741, 173], [0, 350, 62, 475], [68, 158, 132, 346], [383, 142, 435, 225], [45, 120, 95, 175], [145, 105, 195, 160], [210, 105, 260, 160], [285, 105, 335, 165], [350, 105, 400, 165], [520, 90, 570, 150], [575, 90, 625, 150], [650, 90, 700, 150]],
    342: [[241, 209, 381, 424], [122, 226, 250, 396], [0, 206, 131, 402], [364, 210, 541, 443], [361, 176, 450, 294], [251, 154, 365, 299], [125, 169, 257, 297], [641, 142, 766, 423], [355, 110, 418, 168], [520, 106, 575, 166], [222, 125, 285, 180], [547, 101, 625, 233], [304, 125, 354, 176], [697, 90, 748, 140], [718, 112, 815, 260], [55, 105, 105, 165], [105, 100, 155, 160], [165, 105, 215, 165], [430, 95, 480, 155], [480, 92, 530, 150], [580, 90, 630, 155], [645, 85, 695, 145], [760, 85, 815, 150]],
}


def iou(first, second):
    xa, ya = max(first[0], second[0]), max(first[1], second[1])
    xb, yb = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, xb - xa) * max(0, yb - ya)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(1e-9, area_first + area_second - intersection)


def match_metrics(predictions, truth, threshold=0.5):
    candidates = sorted(((iou(prediction, target), p, t) for p, prediction in enumerate(predictions) for t, target in enumerate(truth) if iou(prediction, target) >= threshold), reverse=True)
    used_predictions, used_truth = set(), set()
    for _, prediction_index, truth_index in candidates:
        if prediction_index not in used_predictions and truth_index not in used_truth:
            used_predictions.add(prediction_index)
            used_truth.add(truth_index)
    tp = len(used_predictions)
    fp = len(predictions) - tp
    fn = len(truth) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-9, precision + recall)}


def read_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_number}")
    return frame


def draw(frame, truth, detections, tracks, frame_number, metrics):
    image = frame.copy()
    for box in truth:
        cv2.rectangle(image, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 0, 255), 1)
    for detection in detections:
        box = detection["bbox"]
        cv2.rectangle(image, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 215, 255), 2)
    for track in tracks:
        box = track.bbox
        cv2.rectangle(image, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 255, 0), 2)
        cv2.putText(image, f"ID {track.track_id}", (int(box[0]), max(14, int(box[1]) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
    cv2.putText(image, f"frame {frame_number} | GT {len(truth)} det {len(detections)} active {len(tracks)} | P {metrics['precision']:.2f} R {metrics['recall']:.2f} F1 {metrics['f1']:.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=r"C:\Users\aruna\Downloads\classroom.mp4")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_indices = list(range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max(1, round(fps / 2))))
    all_indices.extend(BENCHMARK_FRAMES)
    all_indices = sorted(set(all_indices))
    detector = TiledYOLOPersonDetector(weights_path="yolov8s.pt", confidence_threshold=0.15, image_size=1280, iou_threshold=0.55, tile_size=640, tile_overlap=0.25, merge_iou=0.55)
    embedder = AnonymousAppearanceEmbedder()
    tracker = DenseByteTracker(track_thresh=0.30, match_thresh=0.80, match_thresh_second=0.40, max_time_lost_frames=12, confirmation_hits=2, center_distance_gate=2.5)
    benchmark = {}
    rendered = []
    track_snapshots = {}
    for frame_number in all_indices:
        frame = read_frame(cap, frame_number)
        detections = detector.detect_persons(frame, frame_number, frame_number / fps)
        for detection in detections:
            detection["appearance"] = embedder(frame, detection["bbox"])
        active = tracker.update(detections, frame_number, frame_number / fps)
        track_snapshots[frame_number] = active
        if frame_number in MANUAL_BOXES:
            metrics = match_metrics([item["bbox"] for item in detections], MANUAL_BOXES[frame_number])
            benchmark[str(frame_number)] = {"manual_students": len(MANUAL_BOXES[frame_number]), "detections": len(detections), "active_tracks": len(active), **metrics}
            annotated = draw(frame, MANUAL_BOXES[frame_number], detections, active, frame_number, metrics)
            cv2.imwrite(str(OUTPUT_DIR / f"manual_benchmark_frame_{frame_number:04d}.jpg"), annotated)
            rendered.append(annotated)
    cap.release()
    writer = cv2.VideoWriter(str(OUTPUT_DIR / "appearance_dense_tracker_benchmark.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (width, height))
    for image in rendered:
        writer.write(image)
    writer.release()
    values = list(benchmark.values())
    total_tp = sum(item["tp"] for item in values)
    total_fp = sum(item["fp"] for item in values)
    total_fn = sum(item["fn"] for item in values)
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    all_tracks = tracker.get_all_observed_tracks()
    report = {"video": args.video, "iou_threshold": 0.5, "manual_annotation_note": "Five fixed frames were manually boxed; boxes cover clearly visible heads/upper bodies and partial edge people.", "benchmark_frames": BENCHMARK_FRAMES, "per_frame": benchmark, "aggregate_detection": {"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-9, precision + recall), "student_coverage": total_tp / max(1, total_tp + total_fn)}, "tracking": {"unique_observed_tracks": len(all_tracks), "peak_active_tracks": max((item["active_tracks"] for item in values), default=0), "mean_active_tracks_on_benchmark_frames": statistics.mean(item["active_tracks"] for item in values), "track_fragmentation_proxy": sum(1 for track in all_tracks if len(track.history) < 3), "track_reappearances_proxy": sum(1 for track in all_tracks if any(b["frame"] - a["frame"] > max(1, round(fps / 2)) for a, b in zip(track.history, track.history[1:]))), "id_switches": "not measurable from anonymous five-frame boxes; no persistent manual identity labels were assigned"}, "artifacts": [f"manual_benchmark_frame_{frame_number:04d}.jpg" for frame_number in BENCHMARK_FRAMES] + ["appearance_dense_tracker_benchmark.mp4"]}
    (OUTPUT_DIR / "manual_dense_tracker_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "manual_dense_tracker_report.md").write_text("# Dense tracker manual benchmark\n\n" + json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()