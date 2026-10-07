import pytest
import os
import json
from scripts.evaluate_ced7_benchmark import (
    box_iou,
    get_spatial_zone,
    get_size_tier,
    load_ced7_annotations,
    aggregate_session_metrics
)

def test_box_iou_calculation():
    boxA = [100.0, 100.0, 200.0, 200.0]
    boxB = [100.0, 100.0, 200.0, 200.0]
    assert box_iou(boxA, boxB) == 1.0

    boxC = [300.0, 300.0, 400.0, 400.0]
    assert box_iou(boxA, boxC) == 0.0

    boxD = [150.0, 100.0, 250.0, 200.0]
    # Inter = 50 * 100 = 5000, Union = 10000 + 10000 - 5000 = 15000 -> IoU = 1/3
    assert abs(box_iou(boxA, boxD) - (1.0 / 3.0)) < 1e-4

def test_spatial_zone_classification():
    # In CDED-7: BACK < 580, MIDDLE <= 750, FRONT > 750
    assert get_spatial_zone(500.0, 1080.0) == "BACK"
    assert get_spatial_zone(650.0, 1080.0) == "MIDDLE"
    assert get_spatial_zone(800.0, 1080.0) == "FRONT"

def test_size_tier_classification():
    # SMALL < 15000, MEDIUM <= 28000, LARGE > 28000
    assert get_size_tier(12000.0) == "SMALL"
    assert get_size_tier(20000.0) == "MEDIUM"
    assert get_size_tier(35000.0) == "LARGE"

def test_ced7_annotations_loading():
    csv_path = "storage/ced7/derived_release/derived_status_boxes.csv"
    if os.path.exists(csv_path):
        data = load_ced7_annotations(csv_path)
        assert "class_1" in data
        assert "class_7" in data
        assert len(data["class_1"]) > 100

def test_aggregate_session_metrics():
    dummy_results = [
        {
            "detection": {
                "total_gt": 100, "total_pred": 100, "total_tp": 85, "total_fp": 15, "total_fn": 15,
                "duplicate_count": 5, "duplicate_rate": 0.05
            },
            "spatial_breakdown": {
                "FRONT": {"gt": 30, "tp": 28, "fp": 2, "fn": 2},
                "MIDDLE": {"gt": 40, "tp": 35, "fp": 5, "fn": 5},
                "BACK": {"gt": 30, "tp": 22, "fp": 8, "fn": 8},
            },
            "size_breakdown": {
                "SMALL": {"gt": 30, "tp": 22, "fp": 8, "fn": 8},
                "MEDIUM": {"gt": 40, "tp": 35, "fp": 5, "fn": 5},
                "LARGE": {"gt": 30, "tp": 28, "fp": 2, "fn": 2},
            },
            "coverage": {
                "detection_coverage_pct": 85.0, "tracking_coverage_pct": 95.0, "temporal_coverage_pct": 80.0
            },
            "tracking": {"fog_recovery": {"stitched_fragments_count": 1, "interpolated_frames_count": 5}},
            "performance": {"latency_ms": 250.0, "processing_fps": 4.0}
        }
    ]
    agg = aggregate_session_metrics(dummy_results)
    assert agg["detection"]["f1"] == 0.85
    assert agg["detection"]["precision"] == 0.85
    assert agg["detection"]["recall"] == 0.85
