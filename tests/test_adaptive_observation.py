import os
import cv2
import numpy as np
import pytest
from app.core.config import settings
from app.ml.detector import (
    AdaptiveHighResObservationDetector,
    HighRecallTiledDetector,
    TiledYOLOPersonDetector,
    YOLOPersonDetector,
)
from app.ml.observation import (
    compute_session_observation_summary,
    compute_track_observation_metrics,
    is_back_row_bbox,
)


def test_is_back_row_bbox():
    """Verify back-row classification thresholding logic."""
    frame_h = 480
    # Box with center y in top 30% of frame -> back row
    top_box = [100, 50, 150, 120]  # y_center = 85 < 480 * 0.45 = 216
    assert is_back_row_bbox(top_box, frame_height=frame_h) is True

    # Box with center y in bottom 40% of frame -> front row
    bottom_box = [100, 300, 200, 450]  # y_center = 375 > 216
    assert is_back_row_bbox(bottom_box, frame_height=frame_h) is False


def test_compute_track_observation_metrics():
    """Verify track-level observation quality, coverage, temporal readiness, and uncertainty."""
    # Synthetic track observed for 20 frames out of 50 total frames
    history = [
        {"frame": f, "timestamp": f * 0.5, "bbox": [100, 50, 160, 130], "confidence": 0.85}
        for f in range(20)
    ]
    track = {
        "track_id": 1,
        "start_frame": 0,
        "end_frame": 19,
        "bounding_box_history": history,
    }

    metrics = compute_track_observation_metrics(
        track=track,
        total_session_frames=50,
        required_sequence_length=16,
        frame_height=480,
    )

    assert metrics["track_id"] == 1
    assert metrics["detection_coverage"] == 0.40  # 20 / 50
    assert metrics["tracking_coverage"] == 0.40   # 20 / 50
    assert metrics["is_temporal_ready"] is True   # 20 >= 16
    assert metrics["temporal_readiness"] == 1.0
    assert metrics["is_back_row"] is True
    assert 0.0 <= metrics["visibility_quality"] <= 1.0
    assert 0.0 <= metrics["uncertainty"] <= 1.0
    assert metrics["observation_state"] in ("CONFIDENT_OBSERVATION", "TEMPORAL_READY", "BEHAVIOUR_UNCERTAIN")


def test_compute_track_observation_metrics_marginal():
    """Verify track with fewer than required frames is tagged as MARGINAL_VISIBILITY."""
    # Only 5 frames observed (less than required 16)
    history = [
        {"frame": f, "timestamp": f * 0.5, "bbox": [20, 20, 50, 60], "confidence": 0.35}
        for f in range(5)
    ]
    track = {
        "track_id": 2,
        "start_frame": 0,
        "end_frame": 4,
        "bounding_box_history": history,
    }

    metrics = compute_track_observation_metrics(
        track=track,
        total_session_frames=100,
        required_sequence_length=16,
        frame_height=480,
    )

    assert metrics["is_temporal_ready"] is False
    assert metrics["temporal_readiness"] < 1.0
    assert metrics["observation_state"] == "MARGINAL_VISIBILITY"


def test_compute_session_observation_summary():
    """Verify room-level aggregation and reference student count handling."""
    tracks = [
        {
            "track_id": 1,
            "start_frame": 0,
            "end_frame": 25,
            "bounding_box_history": [{"frame": f, "bbox": [50, 50, 100, 120], "confidence": 0.8} for f in range(20)],
        },
        {
            "track_id": 2,
            "start_frame": 5,
            "end_frame": 10,
            "bounding_box_history": [{"frame": f, "bbox": [200, 300, 300, 450], "confidence": 0.7} for f in range(5, 10)],
        },
    ]

    summary = compute_session_observation_summary(
        track_results=tracks,
        total_session_frames=30,
        required_sequence_length=16,
        expected_reference_students=70,
    )

    assert summary["total_detected_tracks"] == 2
    assert summary["stable_tracks_count"] == 1       # Track 1 has 20 obs >= 10
    assert summary["temporal_ready_tracks_count"] == 1  # Track 1 has 20 obs >= 16
    assert summary["reference_student_count"] == 70
    assert summary["reference_coverage_ratio"] == round(2 / 70, 4)
    assert "NOT automated ground truth" in summary["reference_note"]


def test_adaptive_highres_detector_initialization():
    """Verify AdaptiveHighResObservationDetector initialization and parameters."""
    detector = AdaptiveHighResObservationDetector(
        weights_path="yolov8s.pt",
        confidence_threshold=0.23,
        adaptive_confidence=0.18,
        image_size=1280,
        back_row_split=0.50,
        tile_size=540,
    )
    assert detector.image_size == 1280
    assert detector.back_row_split == 0.50
    assert detector.adaptive_confidence == 0.18
    assert detector.tile_size == 540


def test_adaptive_highres_detector_detection():
    """Verify AdaptiveHighResObservationDetector on a real image."""
    real_mp4 = os.path.abspath("storage/test_videos/classroom_real_persons.mp4")
    if os.path.exists("bus.jpg"):
        test_frame = cv2.imread("bus.jpg")
    elif os.path.exists(real_mp4):
        cap = cv2.VideoCapture(real_mp4)
        ret, test_frame = cap.read()
        cap.release()
    else:
        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    detector = AdaptiveHighResObservationDetector(
        weights_path="yolov8s.pt",
        confidence_threshold=0.20,
        adaptive_confidence=0.18,
        image_size=640,
        back_row_split=0.50,
        tile_size=400,
    )

    dets = detector.detect_persons(test_frame, frame_idx=0, timestamp=0.0)
    assert isinstance(dets, list)
    for d in dets:
        assert "bbox" in d
        assert len(d["bbox"]) == 4
        assert "confidence" in d
        assert "class_name" in d
        assert d["class_name"] == "person"
