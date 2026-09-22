"""Benchmark comparison: Current TiledYOLOPersonDetector vs New HighRecallTiledDetector.

Evaluates both detectors on the 5 manually annotated classroom benchmark frames.
Computes:
  - Precision, Recall, F1 (IoU >= 0.50)
  - Back-row recall (GT boxes with y < 170)
  - Duplicate / False Positive counts
  - Missed student counts (FN)
Generates side-by-side visual diagnostic frames and a JSON/Markdown report.
Does NOT modify any model weights, checkpoints, trackers, or datasets.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.detector import HighRecallTiledDetector, TiledYOLOPersonDetector


OUTPUT_DIR = Path("storage/dense_classroom_audit")
BENCHMARK_FRAMES = [0, 86, 171, 256, 342]

MANUAL_BOXES = {
    0: [
        [24, 208, 184, 438], [160, 208, 362, 425], [665, 122, 831, 341],
        [614, 115, 695, 235], [36, 149, 170, 304], [170, 102, 237, 167],
        [448, 112, 540, 404], [166, 170, 249, 299], [9, 125, 73, 184],
        [702, 106, 767, 199], [43, 105, 94, 166], [0, 176, 42, 308],
        [112, 127, 166, 177], [824, 133, 848, 216], [394, 116, 467, 328],
        [596, 98, 647, 188], [527, 109, 618, 242], [361, 99, 432, 292],
        [790, 76, 837, 132], [674, 106, 727, 205], [438, 94, 482, 148],
        [0, 95, 30, 155], [95, 95, 145, 155], [245, 90, 305, 165],
        [300, 88, 350, 170], [480, 82, 525, 145], [550, 82, 600, 150],
        [735, 70, 780, 140],
    ],
    86: [
        [0, 217, 184, 466], [721, 138, 848, 302], [440, 128, 520, 245],
        [520, 121, 603, 236], [502, 165, 714, 347], [0, 102, 47, 173],
        [0, 179, 70, 303], [290, 159, 420, 411], [642, 109, 717, 171],
        [659, 163, 778, 317], [376, 122, 446, 250], [165, 104, 215, 170],
        [295, 124, 373, 210], [786, 130, 848, 221], [188, 113, 256, 240],
        [727, 101, 765, 152], [574, 102, 638, 167], [105, 88, 155, 145],
        [260, 85, 315, 150], [330, 85, 380, 150], [405, 90, 455, 150],
        [470, 90, 520, 150], [545, 82, 585, 145], [610, 85, 660, 145],
        [680, 82, 730, 145], [770, 82, 820, 145],
    ],
    171: [
        [258, 179, 489, 372], [183, 150, 270, 272], [269, 140, 350, 257],
        [104, 144, 191, 283], [408, 175, 507, 314], [451, 151, 607, 302],
        [607, 133, 759, 421], [552, 132, 615, 212], [598, 123, 629, 195],
        [8, 178, 146, 465], [12, 149, 98, 248], [392, 125, 452, 183],
        [800, 131, 848, 224], [62, 122, 111, 173], [319, 119, 376, 184],
        [504, 143, 591, 228], [0, 114, 50, 251], [671, 138, 764, 392],
        [150, 105, 200, 160], [220, 95, 270, 155], [285, 95, 335, 155],
        [350, 90, 400, 155], [430, 95, 480, 150], [485, 95, 535, 150],
        [635, 92, 680, 150], [705, 90, 755, 145], [760, 95, 810, 150],
    ],
    256: [
        [404, 183, 634, 373], [412, 145, 493, 261], [253, 150, 340, 291],
        [559, 178, 663, 315], [333, 161, 415, 275], [623, 179, 769, 306],
        [182, 181, 289, 373], [764, 138, 848, 445], [704, 127, 769, 208],
        [530, 127, 600, 197], [735, 114, 788, 206], [608, 119, 649, 182],
        [461, 123, 530, 189], [103, 163, 186, 388], [703, 115, 741, 173],
        [0, 350, 62, 475], [68, 158, 132, 346], [383, 142, 435, 225],
        [45, 120, 95, 175], [145, 105, 195, 160], [210, 105, 260, 160],
        [285, 105, 335, 165], [350, 105, 400, 165], [520, 90, 570, 150],
        [575, 90, 625, 150], [650, 90, 700, 150],
    ],
    342: [
        [241, 209, 381, 424], [122, 226, 250, 396], [0, 206, 131, 402],
        [364, 210, 541, 443], [361, 176, 450, 294], [251, 154, 365, 299],
        [125, 169, 257, 297], [641, 142, 766, 423], [355, 110, 418, 168],
        [520, 106, 575, 166], [222, 125, 285, 180], [547, 101, 625, 233],
        [304, 125, 354, 176], [697, 90, 748, 140], [718, 112, 815, 260],
        [55, 105, 105, 165], [105, 100, 155, 160], [165, 105, 215, 165],
        [430, 95, 480, 155], [480, 92, 530, 150], [580, 90, 630, 155],
        [645, 85, 695, 145], [760, 85, 815, 150],
    ],
}


def iou(first: List[float], second: List[float]) -> float:
    xa, ya = max(first[0], second[0]), max(first[1], second[1])
    xb, yb = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def match_metrics(predictions: List[List[float]], truth: List[List[float]], threshold: float = 0.5) -> Dict[str, Any]:
    candidates = sorted(
        (
            (iou(prediction, target), p, t)
            for p, prediction in enumerate(predictions)
            for t, target in enumerate(truth)
            if iou(prediction, target) >= threshold
        ),
        reverse=True,
    )
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
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def is_back_row(box: List[float]) -> bool:
    """Classify a box as back-row if its vertical center is in the upper region (y < 170)."""
    y_center = (box[1] + box[3]) / 2.0
    return y_center < 170.0


def count_duplicates(boxes: List[List[float]], duplicate_iou: float = 0.50) -> int:
    """Count how many pairs of predicted boxes have high IoU overlap (indicating double-detections)."""
    dups = 0
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            if iou(boxes[i], boxes[j]) >= duplicate_iou:
                dups += 1
    return dups


def read_frame(cap: cv2.VideoCapture, frame_number: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_number}")
    return frame


def draw_panel(
    frame: np.ndarray,
    truth: List[List[float]],
    detections: List[Dict[str, Any]],
    title: str,
    metrics: Dict[str, Any],
    back_row_metrics: Dict[str, Any],
) -> np.ndarray:
    img = frame.copy()
    # Draw GT boxes in thin red
    for box in truth:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)

    # Draw detection boxes in cyan/yellow
    for d in detections:
        box = d["bbox"]
        conf = d["confidence"]
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 220, 255), 2)
        cv2.putText(
            img,
            f"{conf:.2f}",
            (int(box[0]), max(12, int(box[1]) - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 255, 255),
            1,
        )

    # Header banner
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 48), (20, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.82, img, 0.18, 0)

    cv2.putText(
        img,
        title,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 200),
        1,
        cv2.LINE_AA,
    )
    metric_str = (
        f"GT:{len(truth)} Det:{len(detections)} | P:{metrics['precision']:.3f} R:{metrics['recall']:.3f} F1:{metrics['f1']:.3f} "
        f"| BackRow R:{back_row_metrics['recall']:.3f} | FP:{metrics['fp']} FN:{metrics['fn']}"
    )
    cv2.putText(
        img,
        metric_str,
        (8, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def evaluate_detector(
    detector: Any,
    frames: Dict[int, np.ndarray],
) -> Tuple[Dict[str, Any], Dict[int, List[Dict[str, Any]]]]:
    per_frame = {}
    detections_by_frame = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_back_tp = 0
    total_back_fp = 0
    total_back_fn = 0
    total_back_gt = 0

    total_raw_detections = 0
    total_duplicates = 0

    for f_idx in BENCHMARK_FRAMES:
        frame = frames[f_idx]
        dets = detector.detect_persons(frame, frame_idx=f_idx, timestamp=f_idx / 30.0)
        detections_by_frame[f_idx] = dets
        raw_count = getattr(detector, "last_raw_detection_count", len(dets))
        total_raw_detections += raw_count

        gt_boxes = MANUAL_BOXES[f_idx]
        det_boxes = [d["bbox"] for d in dets]
        metrics = match_metrics(det_boxes, gt_boxes)
        dups = count_duplicates(det_boxes, duplicate_iou=0.50)
        total_duplicates += dups

        # Back-row subset
        back_gt = [b for b in gt_boxes if is_back_row(b)]
        back_dets = [b for b in det_boxes if is_back_row(b)]
        back_metrics = match_metrics(back_dets, back_gt)

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        total_back_tp += back_metrics["tp"]
        total_back_fp += back_metrics["fp"]
        total_back_fn += back_metrics["fn"]
        total_back_gt += len(back_gt)

        per_frame[str(f_idx)] = {
            "gt_students": len(gt_boxes),
            "detections": len(dets),
            "raw_detections": raw_count,
            "duplicates": dups,
            "overall": metrics,
            "back_row": {
                "gt": len(back_gt),
                "det": len(back_dets),
                **back_metrics,
            },
        }

    agg_p = total_tp / max(1, total_tp + total_fp)
    agg_r = total_tp / max(1, total_tp + total_fn)
    agg_f1 = 2 * agg_p * agg_r / max(1e-9, agg_p + agg_r)

    back_p = total_back_tp / max(1, total_back_tp + total_back_fp)
    back_r = total_back_tp / max(1, total_back_tp + total_back_fn)
    back_f1 = 2 * back_p * back_r / max(1e-9, back_p + back_r)

    summary = {
        "aggregate": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(agg_p, 4),
            "recall": round(agg_r, 4),
            "f1": round(agg_f1, 4),
            "mean_fp_per_frame": round(total_fp / len(BENCHMARK_FRAMES), 2),
            "mean_fn_per_frame": round(total_fn / len(BENCHMARK_FRAMES), 2),
            "total_duplicates": total_duplicates,
            "total_raw_detections": total_raw_detections,
        },
        "back_row": {
            "total_gt": total_back_gt,
            "tp": total_back_tp,
            "fp": total_back_fp,
            "fn": total_back_fn,
            "precision": round(back_p, 4),
            "recall": round(back_r, 4),
            "f1": round(back_f1, 4),
        },
        "per_frame": per_frame,
    }
    return summary, detections_by_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=r"C:\Users\aruna\Downloads\classroom.mp4")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video at {args.video}")

    frames = {}
    for f_idx in BENCHMARK_FRAMES:
        frames[f_idx] = read_frame(cap, f_idx)
    cap.release()

    print(">>> 1. Evaluating Current TiledYOLOPersonDetector...")
    current_detector = TiledYOLOPersonDetector(
        weights_path="yolov8s.pt",
        confidence_threshold=0.15,
        image_size=1280,
        iou_threshold=0.55,
        tile_size=640,
        tile_overlap=0.25,
        merge_iou=0.55,
    )
    current_results, current_dets = evaluate_detector(current_detector, frames)

    print(">>> 2. Evaluating New HighRecallTiledDetector...")
    highrecall_detector = HighRecallTiledDetector(
        weights_path="yolov8s.pt",
        confidence_threshold=0.23,
        image_size=1280,
        iou_threshold=0.40,
        tile_size=640,
        tile_overlap=0.25,
        fine_tile_size=480,
        fine_overlap=0.40,
        merge_iou=0.40,
        containment_threshold=0.85,
        use_tiles=False,
    )
    highrecall_results, highrecall_dets = evaluate_detector(highrecall_detector, frames)

    # 3. Generate side-by-side diagnostic images
    side_by_side_paths = []
    for f_idx in BENCHMARK_FRAMES:
        gt = MANUAL_BOXES[f_idx]
        cur_panel = draw_panel(
            frames[f_idx],
            gt,
            current_dets[f_idx],
            f"Current TiledYOLO (Frame {f_idx})",
            current_results["per_frame"][str(f_idx)]["overall"],
            current_results["per_frame"][str(f_idx)]["back_row"],
        )
        hr_panel = draw_panel(
            frames[f_idx],
            gt,
            highrecall_dets[f_idx],
            f"New HighRecallTiled (Frame {f_idx})",
            highrecall_results["per_frame"][str(f_idx)]["overall"],
            highrecall_results["per_frame"][str(f_idx)]["back_row"],
        )
        combined = np.hstack([cur_panel, hr_panel])
        out_path = OUTPUT_DIR / f"highrecall_comparison_frame_{f_idx:04d}.jpg"
        cv2.imwrite(str(out_path), combined)
        side_by_side_paths.append(str(out_path))

    # 4. Compile full comparative report
    report = {
        "video": args.video,
        "benchmark_frames": BENCHMARK_FRAMES,
        "current_tiled_detector": current_results,
        "highrecall_tiled_detector": highrecall_results,
        "delta": {
            "precision_diff": round(
                highrecall_results["aggregate"]["precision"] - current_results["aggregate"]["precision"], 4
            ),
            "recall_diff": round(
                highrecall_results["aggregate"]["recall"] - current_results["aggregate"]["recall"], 4
            ),
            "f1_diff": round(
                highrecall_results["aggregate"]["f1"] - current_results["aggregate"]["f1"], 4
            ),
            "back_row_recall_diff": round(
                highrecall_results["back_row"]["recall"] - current_results["back_row"]["recall"], 4
            ),
            "fp_per_frame_diff": round(
                highrecall_results["aggregate"]["mean_fp_per_frame"]
                - current_results["aggregate"]["mean_fp_per_frame"],
                2,
            ),
            "fn_per_frame_diff": round(
                highrecall_results["aggregate"]["mean_fn_per_frame"]
                - current_results["aggregate"]["mean_fn_per_frame"],
                2,
            ),
            "duplicates_diff": (
                highrecall_results["aggregate"]["total_duplicates"]
                - current_results["aggregate"]["total_duplicates"]
            ),
        },
        "side_by_side_artifacts": [p for p in side_by_side_paths],
    }

    report_json_path = OUTPUT_DIR / "highrecall_benchmark_report.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("DETECTOR UPGRADE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} | {'Current Tiled':<15} | {'New HighRecall':<15} | {'Delta':<10}")
    print("-" * 72)
    c_agg = current_results["aggregate"]
    h_agg = highrecall_results["aggregate"]
    d = report["delta"]
    print(f"{'Precision':<25} | {c_agg['precision']:<15.4f} | {h_agg['precision']:<15.4f} | {d['precision_diff']:<+10.4f}")
    print(f"{'Recall':<25} | {c_agg['recall']:<15.4f} | {h_agg['recall']:<15.4f} | {d['recall_diff']:<+10.4f}")
    print(f"{'F1 Score':<25} | {c_agg['f1']:<15.4f} | {h_agg['f1']:<15.4f} | {d['f1_diff']:<+10.4f}")
    c_back = current_results["back_row"]
    h_back = highrecall_results["back_row"]
    print(f"{'Back-Row Recall':<25} | {c_back['recall']:<15.4f} | {h_back['recall']:<15.4f} | {d['back_row_recall_diff']:<+10.4f}")
    print(f"{'Back-Row F1':<25} | {c_back['f1']:<15.4f} | {h_back['f1']:<15.4f} | {h_back['f1'] - c_back['f1']:<+10.4f}")
    print(f"{'Mean FP / Frame':<25} | {c_agg['mean_fp_per_frame']:<15.2f} | {h_agg['mean_fp_per_frame']:<15.2f} | {d['fp_per_frame_diff']:<+10.2f}")
    print(f"{'Mean FN / Frame':<25} | {c_agg['mean_fn_per_frame']:<15.2f} | {h_agg['mean_fn_per_frame']:<15.2f} | {d['fn_per_frame_diff']:<+10.2f}")
    print(f"{'Total Duplicates':<25} | {c_agg['total_duplicates']:<15} | {h_agg['total_duplicates']:<15} | {d['duplicates_diff']:<+10}")
    print("=" * 60)

    # Acceptance test
    f1_improved = d["f1_diff"] > 0
    fp_controlled = d["fp_per_frame_diff"] <= 3.0
    back_row_improved = d["back_row_recall_diff"] > 0
    passed = f1_improved and fp_controlled and back_row_improved
    print(f"Acceptance Criteria (F1 improved: {f1_improved}, FP <= +3: {fp_controlled}, BackRow improved: {back_row_improved}): {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
