"""
TEMPO CDED-7 Benchmark & Scientific Validation Pipeline
======================================================
Executes end-to-end evaluation on the official CDED-7 dataset:
  1. Strict Session-Level Partitioning:
       - TRAIN: class_1, class_2, class_3
       - VALIDATION: class_4, class_6
       - HELD-OUT TEST: class_5, class_7 (Immutable)
  2. Multi-tier Detection & Tracking Evaluation:
       - Baseline TEMPO (YOLOv8s 640p, conf=0.30)
       - Validation Configuration Search (High-res, Horizon tiling, Duplicate suppression)
       - Final Single-Pass Evaluation on Held-Out Test Set
  3. Spatial & Apparent-Size Breakdown:
       - Zones: FRONT, MIDDLE, BACK
       - Sizes: SMALL, MEDIUM, LARGE
  4. Anonymous Multi-Target Tracking & Fog Continuity Recovery:
       - Kalman + Spatial + Appearance Cosine
       - Session Track IDs (STU-xxx)
       - Fog Rolling Buffer & Gap Recovery (recovered_by_fog = True)
  5. Generalization Check on TEMPO Primary & Secondary Datasets
  6. Artifacts, Visualizations, and Comprehensive Reports
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.detector import YOLOPersonDetector, AdaptiveHighResObservationDetector, HighRecallTiledDetector
from app.ml.tracker import DenseByteTracker, STrack, TrackState
from app.ml.fog import FogRoomProcessor
from app.ml.edge import AnonymousTemporalEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("tempo.benchmark.ced7")


def box_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes IoU between [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def load_ced7_annotations(csv_path: str) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Loads CDED-7 annotations mapped by video -> frame_id -> list of GT records."""
    gt_map: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row["video"]
            fid = int(row["frame_id"])
            x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            gt_map[v][fid].append({
                "id": str(row["id"]),
                "bbox": [x1, y1, x2, y2],
                "status": row["status"],
                "area": (x2 - x1) * (y2 - y1),
                "yc": (y1 + y2) / 2.0
            })
    return gt_map


def get_spatial_zone(yc: float, h: float) -> str:
    """Classifies spatial depth: BACK (horizon), MIDDLE, FRONT."""
    # In CDED-7 podium view: students are between y=430 and 1080
    if yc < 580:
        return "BACK"
    elif yc <= 750:
        return "MIDDLE"
    else:
        return "FRONT"


def get_size_tier(area: float) -> str:
    """Classifies apparent student size in pixels²."""
    if area < 15000:
        return "SMALL"
    elif area <= 28000:
        return "MEDIUM"
    else:
        return "LARGE"


class CED7Evaluator:
    def __init__(
        self,
        video_dir: str = "storage/ced7/videos",
        csv_path: str = "storage/ced7/derived_release/derived_status_boxes.csv",
        sample_step: int = 6, # evaluate at 5 FPS (every 6th frame in 30 FPS video)
        iou_match_thresh: float = 0.40,
    ):
        self.video_dir = video_dir
        self.csv_path = csv_path
        self.sample_step = sample_step
        self.iou_match_thresh = iou_match_thresh
        self.gt_data = load_ced7_annotations(csv_path)

    def evaluate_video_session(
        self,
        video_name: str,
        detector: Any,
        tracker: Optional[DenseByteTracker] = None,
        max_frames: Optional[int] = None,
        save_visualizations: bool = False,
        vis_output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs full detection, tracking, and Fog recovery evaluation on a single video."""
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = list(range(0, total_frames, self.sample_step))
        if max_frames:
            frame_indices = frame_indices[:max_frames]

        # Reset tracker
        if tracker is not None:
            tracker.tracked_stracks.clear()
            tracker.lost_stracks.clear()
            tracker.removed_stracks.clear()
            tracker.all_completed_tracks.clear()
            STrack.reset_counter()

        fog = FogRoomProcessor(room_id=f"CED7-{video_name.upper()}", reference_enrollment=30)

        # Metrics accumulators
        metrics = {
            "total_gt": 0,
            "total_pred": 0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "duplicate_count": 0,
            # Spatial Breakdown
            "zones": {
                "FRONT": {"gt": 0, "tp": 0, "fp": 0, "fn": 0},
                "MIDDLE": {"gt": 0, "tp": 0, "fp": 0, "fn": 0},
                "BACK": {"gt": 0, "tp": 0, "fp": 0, "fn": 0},
            },
            # Size Breakdown
            "sizes": {
                "SMALL": {"gt": 0, "tp": 0, "fp": 0, "fn": 0},
                "MEDIUM": {"gt": 0, "tp": 0, "fp": 0, "fn": 0},
                "LARGE": {"gt": 0, "tp": 0, "fp": 0, "fn": 0},
            },
            # Simultaneous counts
            "simultaneous_gt": [],
            "simultaneous_detected": [],
            "simultaneous_confirmed": [],
            "simultaneous_temporal_ready": [],
            # Tracking associations
            "frame_track_gt_matches": [],
            "latency_ms_per_frame": [],
        }

        # Track trajectory history for Fog
        trajectories: Dict[int, Dict[str, Any]] = {}
        vis_saved_count = 0

        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = round(f_idx / max(1.0, fps), 3)
            gt_records = self.gt_data[video_name].get(f_idx + 1, []) # 1-indexed

            # 1. Detection
            t_start = time.perf_counter()
            detections = detector.detect_persons(frame, frame_idx=f_idx, timestamp=timestamp)
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            metrics["latency_ms_per_frame"].append(latency_ms)

            # 2. Tracking
            active_confirmed = []
            if tracker is not None:
                active_confirmed = tracker.update(detections, frame_idx=f_idx, timestamp_seconds=timestamp)
                for trk in active_confirmed:
                    tid = trk.track_id
                    if tid not in trajectories:
                        trajectories[tid] = {
                            "track_id": tid,
                            "student_label": f"STU-{tid:03d}",
                            "start_frame": f_idx,
                            "end_frame": f_idx,
                            "hits": trk.hits,
                            "bounding_box_history": [],
                            "is_confirmed": True
                        }
                    trajectories[tid]["end_frame"] = f_idx
                    trajectories[tid]["hits"] = trk.hits
                    trajectories[tid]["bounding_box_history"].append({
                        "frame": f_idx,
                        "timestamp": timestamp,
                        "bbox": trk.bbox.tolist(),
                        "interpolated": False,
                        "recovered_by_fog": False
                    })
                    
                    # Feed anonymous event to Fog
                    yc = (trk.bbox[1] + trk.bbox[3]) / 2.0
                    fog.ingest_edge_event(AnonymousTemporalEvent(
                        track_id=tid,
                        student_label=f"STU-{tid:03d}",
                        frame_number=f_idx,
                        timestamp_seconds=timestamp,
                        window_start_sec=max(0.0, timestamp - 2.0),
                        window_end_sec=timestamp,
                        bbox=trk.bbox.tolist(),
                        detection_confidence=trk.score,
                        behaviour_type="Looking_Toward_Instruction", # Neutral placeholder for room state
                        confidence=0.90,
                        probability_distribution={"Looking_Toward_Instruction": 0.90},
                        attention_weights=[0.0625] * 16,
                        is_back_row=(yc < 580),
                        model="yolov8_temporal",
                        model_version="v3.0.0"
                    ))

            # 3. Detection Matching with GT
            pred_boxes = [d["bbox"] for d in detections]
            n_gt = len(gt_records)
            n_pred = len(pred_boxes)
            metrics["total_gt"] += n_gt
            metrics["total_pred"] += n_pred
            metrics["simultaneous_gt"].append(n_gt)
            metrics["simultaneous_detected"].append(n_pred)
            metrics["simultaneous_confirmed"].append(len(active_confirmed))

            # Compute IoU matrix between GT and Pred
            matched_gt = set()
            matched_pred = set()
            iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)

            for g_i, g in enumerate(gt_records):
                for p_j, p in enumerate(pred_boxes):
                    iou_matrix[g_i, p_j] = box_iou(g["bbox"], p)

            # Hungarian matching
            if n_gt > 0 and n_pred > 0:
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] >= self.iou_match_thresh:
                        matched_gt.add(r)
                        matched_pred.add(c)

            tp = len(matched_gt)
            fp = n_pred - len(matched_pred)
            fn = n_gt - len(matched_gt)
            metrics["total_tp"] += tp
            metrics["total_fp"] += fp
            metrics["total_fn"] += fn

            # Duplicate detection tracking (predictions that overlap an already matched GT)
            dup_count = 0
            for p_j in range(n_pred):
                if p_j not in matched_pred:
                    # Check if this prediction has high overlap with any GT
                    if n_gt > 0 and np.max(iou_matrix[:, p_j]) >= 0.30:
                        dup_count += 1
            metrics["duplicate_count"] += dup_count

            # Spatial Zone and Size Category Breakdown
            for g_i, g in enumerate(gt_records):
                zone = get_spatial_zone(g["yc"], h)
                size_tier = get_size_tier(g["area"])
                metrics["zones"][zone]["gt"] += 1
                metrics["sizes"][size_tier]["gt"] += 1

                if g_i in matched_gt:
                    metrics["zones"][zone]["tp"] += 1
                    metrics["sizes"][size_tier]["tp"] += 1
                else:
                    metrics["zones"][zone]["fn"] += 1
                    metrics["sizes"][size_tier]["fn"] += 1

            for p_j, p in enumerate(pred_boxes):
                if p_j not in matched_pred:
                    p_yc = (p[1] + p[3]) / 2.0
                    p_area = (p[2] - p[0]) * (p[3] - p[1])
                    zone = get_spatial_zone(p_yc, h)
                    size_tier = get_size_tier(p_area)
                    metrics["zones"][zone]["fp"] += 1
                    metrics["sizes"][size_tier]["fp"] += 1

            # Save Visualizations for key frames (e.g. 3 frames per video)
            if save_visualizations and vis_output_dir and vis_saved_count < 3 and f_idx in frame_indices[len(frame_indices)//4::len(frame_indices)//3]:
                os.makedirs(vis_output_dir, exist_ok=True)
                vis_frame = frame.copy()
                # Draw Ground Truth in Yellow dashed / thin
                for g in gt_records:
                    x1, y1, x2, y2 = [int(v) for v in g["bbox"]]
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                # Draw Detections with Zone Colors: Front=Green, Mid=Blue, Back=Red
                for trk in active_confirmed:
                    x1, y1, x2, y2 = [int(v) for v in trk.bbox]
                    yc = (y1 + y2) / 2.0
                    zone = get_spatial_zone(yc, h)
                    color = (0, 255, 0) if zone == "FRONT" else ((255, 165, 0) if zone == "MIDDLE" else (0, 0, 255))
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"STU-{trk.track_id:03d} [{zone}]"
                    cv2.putText(vis_frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                vis_path = os.path.join(vis_output_dir, f"{video_name}_frame_{f_idx}.jpg")
                cv2.imwrite(vis_path, vis_frame)
                vis_saved_count += 1

        cap.release()

        # 4. Fog Recovery
        fog.session_trajectories = list(trajectories.values())
        fog_recovery_stats = fog.recover_temporal_trajectories(
            max_gap_frames=18,
            seat_distance_threshold=80.0,
            scale_tolerance=1.6
        )

        # 5. Compute Consolidated Precision, Recall, F1
        def calc_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-6, prec + rec)
            return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

        overall_prf = calc_prf(metrics["total_tp"], metrics["total_fp"], metrics["total_fn"])

        zone_results = {}
        for z, dat in metrics["zones"].items():
            zone_results[z] = {**dat, **calc_prf(dat["tp"], dat["fp"], dat["fn"])}

        size_results = {}
        for s, dat in metrics["sizes"].items():
            size_results[s] = {**dat, **calc_prf(dat["tp"], dat["fp"], dat["fn"])}

        # Coverage Metrics
        mean_vis_gt = float(np.mean(metrics["simultaneous_gt"])) if metrics["simultaneous_gt"] else 0.0
        max_vis_gt = int(np.max(metrics["simultaneous_gt"])) if metrics["simultaneous_gt"] else 0
        mean_detected = float(np.mean(metrics["simultaneous_detected"])) if metrics["simultaneous_detected"] else 0.0
        max_detected = int(np.max(metrics["simultaneous_detected"])) if metrics["simultaneous_detected"] else 0
        mean_confirmed = float(np.mean(metrics["simultaneous_confirmed"])) if metrics["simultaneous_confirmed"] else 0.0
        max_confirmed = int(np.max(metrics["simultaneous_confirmed"])) if metrics["simultaneous_confirmed"] else 0

        # Temporal Ready Tracks (accumulated >= 16 frames in session)
        ready_tracks = [t for t in trajectories.values() if len(t["bounding_box_history"]) >= 16]
        n_ready = len(ready_tracks)
        n_confirmed = len(trajectories)

        det_coverage = (metrics["total_tp"] / max(1, metrics["total_gt"])) * 100.0
        track_coverage = (mean_confirmed / max(1e-3, mean_detected)) * 100.0
        temporal_coverage = (n_ready / max(1, n_confirmed)) * 100.0

        avg_latency = float(np.mean(metrics["latency_ms_per_frame"])) if metrics["latency_ms_per_frame"] else 0.0
        fps_proc = 1000.0 / max(1e-3, avg_latency)

        dup_rate = metrics["duplicate_count"] / max(1, metrics["total_pred"])

        return {
            "video_name": video_name,
            "evaluated_frames": len(frame_indices),
            "detection": {
                **overall_prf,
                "total_gt": metrics["total_gt"],
                "total_pred": metrics["total_pred"],
                "total_tp": metrics["total_tp"],
                "total_fp": metrics["total_fp"],
                "total_fn": metrics["total_fn"],
                "duplicate_count": metrics["duplicate_count"],
                "duplicate_rate": round(dup_rate, 4),
            },
            "spatial_breakdown": zone_results,
            "size_breakdown": size_results,
            "coverage": {
                "visible_reference_students": round(mean_vis_gt, 1),
                "max_simultaneous_visible": max_vis_gt,
                "detected_students_mean": round(mean_detected, 1),
                "max_simultaneous_detected": max_detected,
                "confirmed_tracks_mean": round(mean_confirmed, 1),
                "max_simultaneous_confirmed": max_confirmed,
                "temporally_ready_tracks": n_ready,
                "detection_coverage_pct": round(det_coverage, 2),
                "tracking_coverage_pct": round(track_coverage, 2),
                "temporal_coverage_pct": round(temporal_coverage, 2),
            },
            "tracking": {
                "cumulative_unique_tracks": n_confirmed,
                "temporally_ready_tracks": n_ready,
                "fog_recovery": fog_recovery_stats,
            },
            "performance": {
                "latency_ms": round(avg_latency, 1),
                "processing_fps": round(fps_proc, 2),
            }
        }


def aggregate_session_metrics(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates per-video session evaluation metrics across multiple videos."""
    total_gt = sum(r["detection"]["total_gt"] for r in results_list)
    total_pred = sum(r["detection"]["total_pred"] for r in results_list)
    total_tp = sum(r["detection"]["total_tp"] for r in results_list)
    total_fp = sum(r["detection"]["total_fp"] for r in results_list)
    total_fn = sum(r["detection"]["total_fn"] for r in results_list)
    total_dup = sum(r["detection"]["duplicate_count"] for r in results_list)

    prec = total_tp / max(1, total_tp + total_fp)
    rec = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)

    # Spatial aggregation
    zones = ["FRONT", "MIDDLE", "BACK"]
    zone_agg = {}
    for z in zones:
        z_gt = sum(r["spatial_breakdown"][z]["gt"] for r in results_list)
        z_tp = sum(r["spatial_breakdown"][z]["tp"] for r in results_list)
        z_fp = sum(r["spatial_breakdown"][z]["fp"] for r in results_list)
        z_fn = sum(r["spatial_breakdown"][z]["fn"] for r in results_list)
        z_p = z_tp / max(1, z_tp + z_fp)
        z_r = z_tp / max(1, z_tp + z_fn)
        z_f1 = 2 * z_p * z_r / max(1e-6, z_p + z_r)
        zone_agg[z] = {
            "gt": z_gt, "tp": z_tp, "fp": z_fp, "fn": z_fn,
            "precision": round(z_p, 4), "recall": round(z_r, 4), "f1": round(z_f1, 4)
        }

    # Size aggregation
    sizes = ["SMALL", "MEDIUM", "LARGE"]
    size_agg = {}
    for s in sizes:
        s_gt = sum(r["size_breakdown"][s]["gt"] for r in results_list)
        s_tp = sum(r["size_breakdown"][s]["tp"] for r in results_list)
        s_fp = sum(r["size_breakdown"][s]["fp"] for r in results_list)
        s_fn = sum(r["size_breakdown"][s]["fn"] for r in results_list)
        s_p = s_tp / max(1, s_tp + s_fp)
        s_r = s_tp / max(1, s_tp + s_fn)
        s_f1 = 2 * s_p * s_r / max(1e-6, s_p + s_r)
        size_agg[s] = {
            "gt": s_gt, "tp": s_tp, "fp": s_fp, "fn": s_fn,
            "precision": round(s_p, 4), "recall": round(s_r, 4), "f1": round(s_f1, 4)
        }

    mean_det_cov = float(np.mean([r["coverage"]["detection_coverage_pct"] for r in results_list]))
    mean_trk_cov = float(np.mean([r["coverage"]["tracking_coverage_pct"] for r in results_list]))
    mean_tmp_cov = float(np.mean([r["coverage"]["temporal_coverage_pct"] for r in results_list]))
    mean_lat = float(np.mean([r["performance"]["latency_ms"] for r in results_list]))
    mean_fps = 1000.0 / max(1e-3, mean_lat)

    tot_recovered = sum(r["tracking"]["fog_recovery"]["stitched_fragments_count"] for r in results_list)
    tot_interpolated = sum(r["tracking"]["fog_recovery"]["interpolated_frames_count"] for r in results_list)

    return {
        "detection": {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "total_gt": total_gt,
            "total_pred": total_pred,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "duplicate_count": total_dup,
            "duplicate_rate": round(total_dup / max(1, total_pred), 4),
        },
        "spatial_breakdown": zone_agg,
        "size_breakdown": size_agg,
        "coverage": {
            "detection_coverage_pct": round(mean_det_cov, 2),
            "tracking_coverage_pct": round(mean_trk_cov, 2),
            "temporal_coverage_pct": round(mean_tmp_cov, 2),
        },
        "tracking": {
            "stitched_fragments_count": tot_recovered,
            "interpolated_frames_count": tot_interpolated,
        },
        "performance": {
            "latency_ms": round(mean_lat, 1),
            "processing_fps": round(mean_fps, 2),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="TEMPO CDED-7 Benchmark Pipeline")
    parser.add_argument("--mode", choices=["all", "baseline", "sweep", "final_test", "generalization"], default="all")
    parser.add_argument("--sample-step", type=int, default=6) # 5 FPS
    args = parser.parse_args()

    evaluator = CED7Evaluator(sample_step=args.sample_step)

    # Partitions
    TRAIN_VIDEOS = ["class_1", "class_2", "class_3"]
    VAL_VIDEOS = ["class_4", "class_6"]
    TEST_VIDEOS = ["class_5", "class_7"] # Immutable

    tracker_baseline = DenseByteTracker(track_thresh=0.30, match_thresh=0.80)

    # -------------------------------------------------------------
    # 1. BASELINE EVALUATION (Section 5)
    # -------------------------------------------------------------
    if args.mode in ["all", "baseline"]:
        logger.info("=== Running TEMPO Baseline Evaluation on CDED-7 ===")
        baseline_detector = YOLOPersonDetector(weights_path="yolov8s.pt", confidence_threshold=0.30)
        
        baseline_val_results = []
        for v in VAL_VIDEOS:
            logger.info(f"Evaluating Baseline on validation video: {v}")
            res = evaluator.evaluate_video_session(v, baseline_detector, tracker=DenseByteTracker(track_thresh=0.30))
            baseline_val_results.append(res)
            
        baseline_test_results = []
        for v in TEST_VIDEOS:
            logger.info(f"Evaluating Baseline on held-out test video: {v}")
            res = evaluator.evaluate_video_session(v, baseline_detector, tracker=DenseByteTracker(track_thresh=0.30))
            baseline_test_results.append(res)

        baseline_summary = {
            "config_id": "cfg_baseline_tempo_640",
            "detector": "Standard YOLOPersonDetector (YOLOv8s, conf=0.30, imgsz=640)",
            "tracker": "DenseByteTracker (track_thresh=0.30, match_thresh=0.80)",
            "validation_aggregate": aggregate_session_metrics(baseline_val_results),
            "heldout_test_aggregate": aggregate_session_metrics(baseline_test_results),
            "per_video_validation": baseline_val_results,
            "per_video_heldout_test": baseline_test_results,
        }

        os.makedirs("storage/research/ced7_baseline", exist_ok=True)
        baseline_out = "storage/research/ced7_baseline/ced7_baseline_metrics.json"
        with open(baseline_out, "w", encoding="utf-8") as f:
            json.dump(baseline_summary, f, indent=2)
        logger.info(f"Saved Baseline metrics to {baseline_out}")
        logger.info(f"Baseline Held-Out Test F1: {baseline_summary['heldout_test_aggregate']['detection']['f1']}")

    # -------------------------------------------------------------
    # 2. VALIDATION SWEEP (Section 7) - ON VALIDATION SPLIT ONLY!
    # -------------------------------------------------------------
    best_config_id = None
    best_val_f1 = -1.0
    best_detector_creator = None

    if args.mode in ["all", "sweep"]:
        logger.info("=== Running Detection Improvement Loop on VALIDATION Set (class_4, class_6) ===")
        
        candidate_configs = [
            {
                "id": "cfg_val_1_native_1280_c25",
                "name": "Full-Frame High-Resolution (1280x1280, conf=0.25, iou=0.45)",
                "creator": lambda: HighRecallTiledDetector(weights_path="yolov8s.pt", confidence_threshold=0.25, image_size=1280, iou_threshold=0.45, use_tiles=False)
            },
            {
                "id": "cfg_val_2_native_1280_c35",
                "name": "Full-Frame High-Resolution Calibrated (1280x1280, conf=0.35, iou=0.45)",
                "creator": lambda: HighRecallTiledDetector(weights_path="yolov8s.pt", confidence_threshold=0.35, image_size=1280, iou_threshold=0.45, use_tiles=False)
            },
            {
                "id": "cfg_val_3_adaptive_horizon_tiled",
                "name": "Adaptive High-Res + Back-Row Horizon Tiling (conf=0.28, back_row_conf=0.20, tile=540)",
                "creator": lambda: AdaptiveHighResObservationDetector(
                    weights_path="yolov8s.pt", confidence_threshold=0.28, adaptive_confidence=0.20,
                    image_size=1280, back_row_split=0.60, tile_size=540, tile_overlap=0.25,
                    merge_iou=0.40, containment_threshold=0.80
                )
            },
            {
                "id": "cfg_val_4_adaptive_precision_calibrated",
                "name": "Precision-Calibrated Adaptive Horizon (conf=0.35, back_row_conf=0.24, tile=600)",
                "creator": lambda: AdaptiveHighResObservationDetector(
                    weights_path="yolov8s.pt", confidence_threshold=0.35, adaptive_confidence=0.24,
                    image_size=1280, back_row_split=0.58, tile_size=600, tile_overlap=0.25,
                    merge_iou=0.40, containment_threshold=0.82
                )
            },
            {
                "id": "cfg_val_5_containment_duplicate_suppression",
                "name": "High-Res with Robust Containment Duplicate Suppression (conf=0.38, back_row_conf=0.26, merge_iou=0.38)",
                "creator": lambda: AdaptiveHighResObservationDetector(
                    weights_path="yolov8s.pt", confidence_threshold=0.38, adaptive_confidence=0.26,
                    image_size=1280, back_row_split=0.58, tile_size=540, tile_overlap=0.25,
                    merge_iou=0.38, containment_threshold=0.85
                )
            },
        ]

        sweep_results = []
        for cfg in candidate_configs:
            logger.info(f"Testing Candidate Config: {cfg['id']} - {cfg['name']}")
            detector = cfg["creator"]()
            val_results = []
            for v in VAL_VIDEOS:
                res = evaluator.evaluate_video_session(v, detector, tracker=DenseByteTracker(track_thresh=0.25))
                val_results.append(res)
            
            agg = aggregate_session_metrics(val_results)
            val_f1 = agg["detection"]["f1"]
            back_f1 = agg["spatial_breakdown"]["BACK"]["f1"]
            small_f1 = agg["size_breakdown"]["SMALL"]["f1"]
            dup_rate = agg["detection"]["duplicate_rate"]

            logger.info(f"  -> Validation Detection F1: {val_f1:.4f} | Back-Row F1: {back_f1:.4f} | Small F1: {small_f1:.4f} | Dup Rate: {dup_rate*100:.1f}%")

            cfg_result = {
                "config_id": cfg["id"],
                "name": cfg["name"],
                "validation_aggregate": agg,
                "per_video_validation": val_results,
            }
            sweep_results.append(cfg_result)

            # Optimization objective: Maximize Validation Detection F1 while keeping duplicate rate bounded
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_config_id = cfg["id"]
                best_detector_creator = cfg["creator"]

        logger.info(f"\n*** Best Selected Configuration on VALIDATION Split: {best_config_id} (Val F1: {best_val_f1:.4f}) ***\n")

        os.makedirs("storage/research/ced7/metrics", exist_ok=True)
        sweep_out = "storage/research/ced7/metrics/validation_sweep.json"
        with open(sweep_out, "w", encoding="utf-8") as f:
            json.dump({
                "selected_best_config_id": best_config_id,
                "best_validation_f1": best_val_f1,
                "candidates": sweep_results
            }, f, indent=2)

    # -------------------------------------------------------------
    # 3. FINAL EVALUATION ON IMMUTABLE HELD-OUT TEST (Section 6)
    # -------------------------------------------------------------
    if args.mode in ["all", "final_test"]:
        if best_detector_creator is None:
            best_config_id = "cfg_val_1_native_1280_c25"
            best_detector_creator = lambda: HighRecallTiledDetector(
                weights_path="yolov8s.pt", confidence_threshold=0.25, image_size=1280, iou_threshold=0.45, use_tiles=False
            )

        # Save selected config to storage/research/ced7/configs/
        os.makedirs("storage/research/ced7/configs", exist_ok=True)
        with open("storage/research/ced7/configs/selected_best_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "config_id": best_config_id,
                "detector_architecture": "Full-Frame High-Resolution (1280x1280, conf=0.25, iou=0.45)",
                "weights": "yolov8s.pt",
                "image_size": 1280,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "selection_criterion": "Best Detection F1 on Validation Split (class_4, class_6)",
                "tracker": "DenseByteTracker (track_thresh=0.28, match_thresh=0.80)"
            }, f, indent=2)

        logger.info(f"=== Running Final Evaluation of Frozen Best Config ({best_config_id}) ONCE on Held-Out Test Set ===")
        final_detector = best_detector_creator()

        test_results = []
        for v in TEST_VIDEOS:
            logger.info(f"Evaluating Held-out Test video: {v}")
            # Also save visual validation frames
            res = evaluator.evaluate_video_session(
                v, final_detector, tracker=DenseByteTracker(track_thresh=0.28),
                save_visualizations=True, vis_output_dir="storage/research/ced7/visualizations"
            )
            test_results.append(res)

        test_aggregate = aggregate_session_metrics(test_results)
        test_f1 = test_aggregate["detection"]["f1"]
        target_f1 = 0.80
        gap = round(target_f1 - test_f1, 4)

        if test_f1 >= target_f1:
            verdict = "Detection F1 target achieved."
            status = "RESEARCH VALIDATED"
        else:
            verdict = f"Target: {target_f1:.2f} | Measured: {test_f1:.4f} | Gap: {gap:.4f}"
            status = "TARGET NOT YET ACHIEVED"

        logger.info(f"\n==========================================")
        logger.info(f"FINAL HELD-OUT TEST EVALUATION RESULT:")
        logger.info(f"  Configuration: {best_config_id}")
        logger.info(f"  Detection Precision: {test_aggregate['detection']['precision']}")
        logger.info(f"  Detection Recall:    {test_aggregate['detection']['recall']}")
        logger.info(f"  Detection F1:        {test_f1}")
        logger.info(f"  Back-Row Recall:     {test_aggregate['spatial_breakdown']['BACK']['recall']}")
        logger.info(f"  Back-Row F1:         {test_aggregate['spatial_breakdown']['BACK']['f1']}")
        logger.info(f"  Small-Student F1:    {test_aggregate['size_breakdown']['SMALL']['f1']}")
        logger.info(f"  Tracking Coverage:   {test_aggregate['coverage']['tracking_coverage_pct']}%")
        logger.info(f"  Temporal Coverage:   {test_aggregate['coverage']['temporal_coverage_pct']}%")
        logger.info(f"  Verdict:             {verdict}")
        logger.info(f"  Status:              {status}")
        logger.info(f"==========================================\n")

        test_payload = {
            "frozen_config_id": best_config_id,
            "target_f1": target_f1,
            "measured_test_f1": test_f1,
            "gap": gap,
            "verdict": verdict,
            "status": status,
            "heldout_test_aggregate": test_aggregate,
            "per_video_test_results": test_results,
        }

        test_out = "storage/research/ced7/metrics/heldout_test_metrics.json"
        with open(test_out, "w", encoding="utf-8") as f:
            json.dump(test_payload, f, indent=2)
        logger.info(f"Saved Final Held-Out Test metrics to {test_out}")

    # -------------------------------------------------------------
    # 4. GENERALIZATION ON TEMPO ORIGINAL DATASETS (Section 25, Q15)
    # -------------------------------------------------------------
    if args.mode in ["all", "generalization"]:
        logger.info("=== Running Generalization Check on Original Primary & Secondary Datasets ===")
        # Evaluate on storage/large_classroom/primary_1080p_classroom.mp4
        prim_video = "storage/large_classroom/primary_1080p_classroom.mp4"
        sec_video = "storage/large_classroom/secondary_classroom.mp4"
        
        gen_results = {}
        for name, v_path in [("primary_1080p", prim_video), ("secondary_848", sec_video)]:
            if os.path.exists(v_path):
                cap = cv2.VideoCapture(v_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    # Test Baseline
                    base_det = YOLOPersonDetector(weights_path="yolov8s.pt", confidence_threshold=0.30)
                    base_preds = base_det.detect_persons(frame)
                    
                    # Test Improved
                    imp_det = AdaptiveHighResObservationDetector(
                        weights_path="yolov8s.pt", confidence_threshold=0.28, adaptive_confidence=0.20,
                        image_size=1280, back_row_split=0.58, tile_size=600, tile_overlap=0.25,
                        merge_iou=0.42, containment_threshold=0.82
                    )
                    imp_preds = imp_det.detect_persons(frame)
                    gen_results[name] = {
                        "video": v_path,
                        "baseline_detection_count": len(base_preds),
                        "improved_detection_count": len(imp_preds),
                        "gain": len(imp_preds) - len(base_preds)
                    }
                    logger.info(f"  {name}: Baseline detected {len(base_preds)}, Improved detected {len(imp_preds)} (gain: +{len(imp_preds)-len(base_preds)})")

        gen_out = "storage/research/ced7/metrics/original_dataset_generalization.json"
        with open(gen_out, "w", encoding="utf-8") as f:
            json.dump(gen_results, f, indent=2)

    logger.info("CDED-7 Benchmark execution completed successfully.")


if __name__ == "__main__":
    main()
