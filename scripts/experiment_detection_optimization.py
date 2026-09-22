"""Experimental harness for final detection optimization.

Tests multi-scale / adaptive back-row crops, input resolutions (1280, 1344, 1408, 1536),
confidence thresholds, NMS IoU, and aspect ratio filtering on the 5 benchmark frames.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_highrecall_detector import (
    MANUAL_BOXES,
    BENCHMARK_FRAMES,
    match_metrics,
    is_back_row,
    read_frame,
    count_duplicates,
    iou,
)

cap = cv2.VideoCapture(r"C:\Users\aruna\Downloads\classroom.mp4")
frames = {f: read_frame(cap, f) for f in BENCHMARK_FRAMES}
cap.release()

model = YOLO("yolov8s.pt")

results = []

def eval_config(
    name,
    imgsz=1280,
    conf=0.22,
    nms_iou=0.45,
    min_aspect=0.17,
    max_aspect=2.5,
    back_row_strip=False,
    strip_conf=0.25,
    strip_h=220,
    merge_iou=0.45,
):
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    back_tp, back_fp, back_fn = 0, 0, 0
    tot_dups = 0

    for f_idx in BENCHMARK_FRAMES:
        frame = frames[f_idx]
        h, w = frame.shape[:2]

        # Pass 1: Full frame
        res = model.predict(
            source=frame,
            classes=[0],
            conf=conf,
            imgsz=imgsz,
            iou=nms_iou,
            verbose=False,
        )[0]
        raw_boxes = []
        if res.boxes is not None:
            for b in res.boxes:
                if int(b.cls[0].cpu().numpy()) == 0:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    bw = xyxy[2] - xyxy[0]
                    bh = xyxy[3] - xyxy[1]
                    ratio = bw / max(1e-3, bh)
                    if min_aspect <= ratio <= max_aspect:
                        raw_boxes.append({
                            "bbox": xyxy,
                            "conf": float(b.conf[0].cpu().numpy()),
                        })

        # Optional Pass 2: Adaptive Back-row Upper Crop
        if back_row_strip:
            strip = frame[0:strip_h, :]
            s_res = model.predict(
                source=strip,
                classes=[0],
                conf=strip_conf,
                imgsz=imgsz,
                iou=nms_iou,
                verbose=False,
            )[0]
            if s_res.boxes is not None:
                for b in s_res.boxes:
                    if int(b.cls[0].cpu().numpy()) == 0:
                        xyxy = b.xyxy[0].cpu().numpy().tolist()
                        bw = xyxy[2] - xyxy[0]
                        bh = xyxy[3] - xyxy[1]
                        ratio = bw / max(1e-3, bh)
                        if min_aspect <= ratio <= max_aspect:
                            raw_boxes.append({
                                "bbox": [xyxy[0], xyxy[1], xyxy[2], xyxy[3]],
                                "conf": float(b.conf[0].cpu().numpy()),
                            })

        # Merge
        rem = sorted(raw_boxes, key=lambda x: x["conf"], reverse=True)
        final_boxes = []
        while rem:
            seed = rem.pop(0)
            group = [seed]
            keep = []
            for c in rem:
                if iou(seed["bbox"], c["bbox"]) >= merge_iou:
                    group.append(c)
                else:
                    keep.append(c)
            rem = keep
            weights = [max(0.01, d["conf"]) for d in group]
            tw = sum(weights)
            merged_bbox = [
                sum(d["bbox"][ax] * weights[i] for i, d in enumerate(group)) / tw
                for ax in range(4)
            ]
            final_boxes.append(merged_bbox)

        gt = MANUAL_BOXES[f_idx]
        m = match_metrics(final_boxes, gt)
        dups = count_duplicates(final_boxes, duplicate_iou=0.50)

        back_gt = [b for b in gt if is_back_row(b)]
        back_dets = [b for b in final_boxes if is_back_row(b)]
        bm = match_metrics(back_dets, back_gt)

        tot_tp += m["tp"]
        tot_fp += m["fp"]
        tot_fn += m["fn"]
        back_tp += bm["tp"]
        back_fp += bm["fp"]
        back_fn += bm["fn"]
        tot_dups += dups

    p = tot_tp / max(1, tot_tp + tot_fp)
    r = tot_tp / max(1, tot_tp + tot_fn)
    f1 = 2 * p * r / max(1e-9, p + r)

    bp = back_tp / max(1, back_tp + back_fp)
    br = back_tp / max(1, back_tp + back_fn)
    bf1 = 2 * bp * br / max(1e-9, bp + br)

    entry = {
        "name": name,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "back_recall": round(br, 4),
        "back_f1": round(bf1, 4),
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "mean_fp": round(tot_fp / len(BENCHMARK_FRAMES), 2),
        "duplicates": tot_dups,
    }
    results.append(entry)
    print(
        f"{name:<42} | P:{p:.4f} R:{r:.4f} F1:{f1:.4f} | Back-R:{br:.4f} Back-F1:{bf1:.4f} | "
        f"TP:{tot_tp:2d} FP:{tot_fp:2d} (avg {entry['mean_fp']:.1f}) FN:{tot_fn:2d} Dup:{tot_dups}"
    )

print("=" * 105)
print("RUNNING FINAL DETECTION OPTIMIZATION EXPERIMENTS")
print("=" * 105)

# Current baseline & highrecall reference
eval_config("Baseline (Tiled 640px, conf 0.15)", imgsz=1280, conf=0.15, nms_iou=0.55) # approx baseline
eval_config("Validated HighRecall (1280px, conf 0.22, iou 0.45)", imgsz=1280, conf=0.22, nms_iou=0.45)

# 1. Confidence fine sweeps around 0.22-0.25
for c in [0.20, 0.21, 0.22, 0.23, 0.24, 0.25]:
    eval_config(f"Full 1280px | conf {c:.2f} | iou 0.45", imgsz=1280, conf=c, nms_iou=0.45)

# 2. NMS IoU sweeps at conf 0.23 and 0.24
for iou_val in [0.38, 0.40, 0.42, 0.45, 0.48]:
    eval_config(f"Full 1280px | conf 0.23 | iou {iou_val:.2f}", imgsz=1280, conf=0.23, nms_iou=iou_val)

# 3. Resolution sweeps (1280 vs 1344 vs 1408)
for res in [1280, 1344, 1408]:
    eval_config(f"Full {res}px | conf 0.23 | iou 0.42", imgsz=res, conf=0.23, nms_iou=0.42)

# 4. Aspect ratio tightness (reject furniture)
eval_config("Full 1280px | conf 0.23 | aspect 0.25-1.5", imgsz=1280, conf=0.23, nms_iou=0.42, min_aspect=0.25, max_aspect=1.50)
eval_config("Full 1280px | conf 0.23 | aspect 0.25-1.3", imgsz=1280, conf=0.23, nms_iou=0.42, min_aspect=0.25, max_aspect=1.30)

# 5. Adaptive Back-row Upper Strip
for s_conf in [0.25, 0.28, 0.30]:
    eval_config(f"1280px + Strip(220px, conf {s_conf})", imgsz=1280, conf=0.23, nms_iou=0.42, back_row_strip=True, strip_conf=s_conf)

print("=" * 105)
