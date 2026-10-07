import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("tempo.ml.observation")


def is_back_row_bbox(bbox: List[float], frame_height: Optional[float] = None, split_ratio: float = 0.45) -> bool:
    """
    Determines whether a bounding box [x1, y1, x2, y2] belongs to the classroom back-row region.
    If frame_height is provided, uses vertical center y_center < frame_height * split_ratio.
    Otherwise, if coordinates are normalized in [0, 1], checks y_center < split_ratio.
    If absolute coordinates without frame_height, uses fallback threshold y_center < 180.0.
    """
    if not bbox or len(bbox) != 4:
        return False
    y_center = (bbox[1] + bbox[3]) / 2.0
    if frame_height is not None and frame_height > 0:
        return y_center < (frame_height * split_ratio)
    if 0.0 <= bbox[3] <= 1.0:
        return y_center < split_ratio
    return y_center < 180.0


def compute_track_observation_metrics(
    track: Dict[str, Any],
    total_session_frames: int,
    required_sequence_length: int = 16,
    frame_height: Optional[float] = None,
    behaviour_predictions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Calculates observation-aware quality metrics for an anonymous student track:
      1. Detection Coverage: ratio of sampled frames with valid detections.
      2. Tracking Coverage: fraction of session lifespan spanned by track.
      3. Temporal Readiness: whether track has >= required_sequence_length observations.
      4. Visibility Quality: score in [0.0, 1.0] reflecting pixel area, confidence, and regularity.
      5. Uncertainty: uncertainty score reflecting detection ambiguity and behaviour entropy.
      6. Observation State: discrete state distinguishing not detected, marginal, temporal ready, etc.
    """
    history = track.get("bounding_box_history", [])
    num_obs = len(history)
    total_frames = max(1, total_session_frames)

    # 1. Detection Coverage
    detection_coverage = min(1.0, num_obs / total_frames)

    # 2. Tracking Coverage
    start_frame = track.get("start_frame", history[0]["frame"] if history else 0)
    end_frame = track.get("end_frame", history[-1]["frame"] if history else 0)
    lifespan = max(1, end_frame - start_frame + 1)
    tracking_coverage = min(1.0, lifespan / total_frames)

    # 3. Temporal Readiness
    readiness_ratio = min(1.0, num_obs / max(1, required_sequence_length))
    is_temporal_ready = (num_obs >= required_sequence_length)

    # 4. Visibility / Observation Quality
    areas = []
    confs = []
    aspect_ratios = []
    back_row_votes = 0

    for h in history:
        box = h.get("bbox", [])
        if len(box) == 4:
            w = max(1e-3, box[2] - box[0])
            h_box = max(1e-3, box[3] - box[1])
            areas.append(w * h_box)
            aspect_ratios.append(w / h_box)
            if is_back_row_bbox(box, frame_height):
                back_row_votes += 1
        conf = h.get("confidence")
        if conf is not None:
            confs.append(float(conf))

    mean_area = float(np.mean(areas)) if areas else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.50
    is_back_row = (back_row_votes > (len(history) / 2)) if history else False

    # Normalized area score: small back-row crops (< 2500 px^2) have lower visibility
    # Front-row crops (> 25000 px^2) have score near 1.0
    area_score = float(np.clip((np.sqrt(mean_area) - 25.0) / 125.0, 0.15, 1.0))

    # Aspect ratio stability: regular human ratio ~ 0.35 - 1.2
    ratio_score = 1.0
    if aspect_ratios:
        mean_ratio = float(np.mean(aspect_ratios))
        if mean_ratio < 0.25 or mean_ratio > 1.8:
            ratio_score = 0.5
        elif mean_ratio < 0.35 or mean_ratio > 1.4:
            ratio_score = 0.8

    visibility_quality = round(
        float(0.40 * area_score + 0.40 * mean_conf + 0.20 * ratio_score), 4
    )

    # 5. Behaviour Uncertainty
    # If predictions are associated with this track, incorporate model confidence
    track_id = track.get("track_id")
    track_preds = [
        p for p in (behaviour_predictions or [])
        if p.get("track_id") == track_id
    ]

    if track_preds:
        pred_confs = [p.get("confidence", 0.5) for p in track_preds]
        mean_pred_conf = float(np.mean(pred_confs))
        uncertainty = round(float(1.0 - (0.5 * visibility_quality + 0.5 * mean_pred_conf)), 4)
    else:
        uncertainty = round(float(1.0 - visibility_quality), 4)

    # 6. Observation State Determination
    if num_obs == 0:
        observation_state = "NOT_DETECTED"
    elif not is_temporal_ready or visibility_quality < 0.35:
        observation_state = "MARGINAL_VISIBILITY"
    elif is_temporal_ready and uncertainty > 0.55:
        observation_state = "BEHAVIOUR_UNCERTAIN"
    elif is_temporal_ready:
        observation_state = "CONFIDENT_OBSERVATION"
    else:
        observation_state = "TEMPORAL_READY"

    return {
        "track_id": track_id,
        "detection_coverage": round(detection_coverage, 4),
        "tracking_coverage": round(tracking_coverage, 4),
        "temporal_readiness": round(readiness_ratio, 4),
        "is_temporal_ready": is_temporal_ready,
        "visibility_quality": visibility_quality,
        "uncertainty": uncertainty,
        "is_back_row": is_back_row,
        "mean_pixel_area": round(mean_area, 1),
        "mean_detection_confidence": round(mean_conf, 4),
        "observed_frames_count": num_obs,
        "observation_state": observation_state,
    }


def compute_session_observation_summary(
    track_results: List[Dict[str, Any]],
    total_session_frames: int,
    required_sequence_length: int = 16,
    expected_reference_students: Optional[int] = None,
    behaviour_predictions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Computes room/session-level aggregated observation quality metrics.
    Clearly distinguishes reference student counts from automated detected ground truth.
    Never fabricates observations or claims all students are visible.
    """
    per_track_metrics = [
        compute_track_observation_metrics(
            t,
            total_session_frames=total_session_frames,
            required_sequence_length=required_sequence_length,
            behaviour_predictions=behaviour_predictions,
        )
        for t in track_results
    ]

    total_detected = len(track_results)
    stable_tracks = [m for m in per_track_metrics if m["observed_frames_count"] >= 10]
    temporal_ready = [m for m in per_track_metrics if m["is_temporal_ready"]]
    back_row_tracks = [m for m in per_track_metrics if m["is_back_row"]]

    mean_det_cov = float(np.mean([m["detection_coverage"] for m in per_track_metrics])) if per_track_metrics else 0.0
    mean_trk_cov = float(np.mean([m["tracking_coverage"] for m in per_track_metrics])) if per_track_metrics else 0.0
    mean_vis_qual = float(np.mean([m["visibility_quality"] for m in per_track_metrics])) if per_track_metrics else 0.0
    mean_uncert = float(np.mean([m["uncertainty"] for m in per_track_metrics])) if per_track_metrics else 0.0

    warnings: List[str] = []
    if mean_vis_qual < 0.40:
        warnings.append("Low average visibility quality across observed students (< 0.40). Camera angle or lighting may be suboptimal.")
    if len(back_row_tracks) == 0 and total_detected > 0:
        warnings.append("No back-row students detected. Back-row seating zone may be occluded or out of frame.")
    if back_row_tracks:
        back_row_vis = float(np.mean([m["visibility_quality"] for m in back_row_tracks]))
        if back_row_vis < 0.35:
            warnings.append(f"Back-row student observation quality is low ({back_row_vis:.2f}). Consider adaptive zoom or high-resolution camera.")

    summary: Dict[str, Any] = {
        "total_detected_tracks": total_detected,
        "stable_tracks_count": len(stable_tracks),
        "temporal_ready_tracks_count": len(temporal_ready),
        "back_row_tracks_count": len(back_row_tracks),
        "mean_detection_coverage": round(mean_det_cov, 4),
        "mean_tracking_coverage": round(mean_trk_cov, 4),
        "mean_visibility_quality": round(mean_vis_qual, 4),
        "mean_uncertainty": round(mean_uncert, 4),
        "observation_warnings": warnings,
        "per_track_observation_metrics": per_track_metrics,
    }

    # If reference/nominal student count provided (e.g. 70 students registered):
    if expected_reference_students is not None and expected_reference_students > 0:
        summary["reference_student_count"] = expected_reference_students
        summary["reference_coverage_ratio"] = round(min(1.0, total_detected / expected_reference_students), 4)
        summary["reference_note"] = (
            f"Reference count ({expected_reference_students}) is nominal/expected enrollment, "
            f"NOT automated ground truth. Automated detection observed {total_detected} unique tracks."
        )

    return summary
