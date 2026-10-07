import cv2
import json
import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from app.ml.detector import YOLOPersonDetector, HighRecallTiledDetector
from app.ml.tracker import DenseByteTracker, STrack

# Load Baseline & Held-Out Test Metrics
with open("storage/research/ced7_baseline/ced7_baseline_metrics.json", "r", encoding="utf-8") as f:
    base_data = json.load(f)

with open("storage/research/ced7/metrics/heldout_test_metrics.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

b_test = base_data["heldout_test_aggregate"]
i_test = test_data["heldout_test_aggregate"]

# Generate comparison JSON
comparison_metrics = {
    "evaluation_split": "Immutable Held-Out CDED-7 Test Set (class_5, class_7)",
    "baseline_config": base_data["detector"],
    "improved_config": test_data["frozen_config_id"],
    "comparison_table": {
        "Detection Precision": {
            "before": b_test["detection"]["precision"],
            "after": i_test["detection"]["precision"],
            "delta": round(i_test["detection"]["precision"] - b_test["detection"]["precision"], 4)
        },
        "Detection Recall": {
            "before": b_test["detection"]["recall"],
            "after": i_test["detection"]["recall"],
            "delta": round(i_test["detection"]["recall"] - b_test["detection"]["recall"], 4)
        },
        "Detection F1": {
            "before": b_test["detection"]["f1"],
            "after": i_test["detection"]["f1"],
            "delta": round(i_test["detection"]["f1"] - b_test["detection"]["f1"], 4)
        },
        "Back-Row Recall": {
            "before": b_test["spatial_breakdown"]["BACK"]["recall"],
            "after": i_test["spatial_breakdown"]["BACK"]["recall"],
            "delta": round(i_test["spatial_breakdown"]["BACK"]["recall"] - b_test["spatial_breakdown"]["BACK"]["recall"], 4)
        },
        "Back-Row F1": {
            "before": b_test["spatial_breakdown"]["BACK"]["f1"],
            "after": i_test["spatial_breakdown"]["BACK"]["f1"],
            "delta": round(i_test["spatial_breakdown"]["BACK"]["f1"] - b_test["spatial_breakdown"]["BACK"]["f1"], 4)
        },
        "Small-Person Recall": {
            "before": b_test["size_breakdown"]["SMALL"]["recall"],
            "after": i_test["size_breakdown"]["SMALL"]["recall"],
            "delta": round(i_test["size_breakdown"]["SMALL"]["recall"] - b_test["size_breakdown"]["SMALL"]["recall"], 4)
        },
        "Small-Person F1": {
            "before": b_test["size_breakdown"]["SMALL"]["f1"],
            "after": i_test["size_breakdown"]["SMALL"]["f1"],
            "delta": round(i_test["size_breakdown"]["SMALL"]["f1"] - b_test["size_breakdown"]["SMALL"]["f1"], 4)
        },
        "Duplicate Rate": {
            "before": b_test["detection"]["duplicate_rate"],
            "after": i_test["detection"]["duplicate_rate"],
            "delta": round(i_test["detection"]["duplicate_rate"] - b_test["detection"]["duplicate_rate"], 4)
        },
        "False Positives": {
            "before": b_test["detection"]["total_fp"],
            "after": i_test["detection"]["total_fp"],
            "delta": i_test["detection"]["total_fp"] - b_test["detection"]["total_fp"]
        },
        "Tracking Coverage": {
            "before": f"{b_test['coverage']['tracking_coverage_pct']}%",
            "after": f"{i_test['coverage']['tracking_coverage_pct']}%",
            "delta": f"{i_test['coverage']['tracking_coverage_pct'] - b_test['coverage']['tracking_coverage_pct']:.2f}%"
        },
        "Temporal Coverage": {
            "before": f"{b_test['coverage']['temporal_coverage_pct']}%",
            "after": f"{i_test['coverage']['temporal_coverage_pct']}%",
            "delta": f"{i_test['coverage']['temporal_coverage_pct'] - b_test['coverage']['temporal_coverage_pct']:.2f}%"
        },
        "Processing FPS": {
            "before": b_test["performance"]["processing_fps"],
            "after": i_test["performance"]["processing_fps"],
            "delta": round(i_test["performance"]["processing_fps"] - b_test["performance"]["processing_fps"], 2)
        }
    }
}

comp_path = "storage/research/ced7/comparisons/before_after_comparison.json"
os.makedirs("storage/research/ced7/comparisons", exist_ok=True)
with open(comp_path, "w", encoding="utf-8") as f:
    json.dump(comparison_metrics, f, indent=2)
print(f"Saved comparison to {comp_path}")

# Generate side-by-side frames:
# We will compare class_7 frame 70 and class_5 frame 110
base_det = YOLOPersonDetector(weights_path="yolov8s.pt", confidence_threshold=0.30)
imp_det = HighRecallTiledDetector(weights_path="yolov8s.pt", confidence_threshold=0.25, image_size=1280, iou_threshold=0.45, use_tiles=False)

vis_dir = "storage/research/ced7/visualizations"
os.makedirs(vis_dir, exist_ok=True)

test_frames = [
    ("class_7", 70),
    ("class_5", 110),
]

for vname, f_idx in test_frames:
    vpath = f"storage/ced7/videos/{vname}.mp4"
    cap = cv2.VideoCapture(vpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret: continue
    
    h, w = frame.shape[:2]
    
    # 1. Baseline Run
    b_preds = base_det.detect_persons(frame, frame_idx=f_idx)
    b_canvas = frame.copy()
    cv2.putText(b_canvas, f"CURRENT TEMPO BASELINE (640p) - {len(b_preds)} detections", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    for p in b_preds:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        yc = (y1 + y2) / 2.0
        # Color red for duplicates/unfiltered
        cv2.rectangle(b_canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(b_canvas, f"det {p['confidence']:.2f}", (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # 2. Improved Run
    i_preds = imp_det.detect_persons(frame, frame_idx=f_idx)
    i_canvas = frame.copy()
    cv2.putText(i_canvas, f"IMPROVED TEMPO (1280p Calibrated) - {len(i_preds)} detections", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    for idx, p in enumerate(i_preds):
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        yc = (y1 + y2) / 2.0
        zone = "BACK" if yc < 580 else ("MIDDLE" if yc <= 750 else "FRONT")
        color = (0, 255, 0) if zone == "FRONT" else ((255, 165, 0) if zone == "MIDDLE" else (0, 255, 255))
        cv2.rectangle(i_canvas, (x1, y1), (x2, y2), color, 2)
        label = f"STU-{idx+1:03d} [{zone}]"
        cv2.putText(i_canvas, label, (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Combine side-by-side (scale down by half for convenient viewing)
    thumb_w, thumb_h = w // 2, h // 2
    b_thumb = cv2.resize(b_canvas, (thumb_w, thumb_h))
    i_thumb = cv2.resize(i_canvas, (thumb_w, thumb_h))
    
    side_by_side = np.hstack([b_thumb, i_thumb])
    out_img = os.path.join(vis_dir, f"side_by_side_comparison_{vname}_f{f_idx}.jpg")
    cv2.imwrite(out_img, side_by_side)
    print(f"Saved side-by-side visualization: {out_img}")
