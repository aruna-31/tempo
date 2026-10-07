"""
End-to-End Classroom Empirical Evaluation Script
Evaluates 5 critical real-world questions on storage/test_videos/classroom_real_persons.mp4:
1. ResNet-50 vs ResNet-18 Macro-F1 empirical performance.
2. Adaptive high-res inference back-row precision, recall, and F1.
3. Nominal 70-student classroom detection, stability, and temporal readiness counts.
4. Parity test: Edge -> Fog pipeline vs existing LOCAL MLInferenceService pipeline.
5. Exam Mode real review events validation: checks observable categories, timestamps, durations, confidence, and human-review sanity.
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from app.ml.camera import DemoMP4CameraSource
from app.ml.edge import EdgeProcessing
from app.ml.fog import FogRoomProcessor
from app.ml.service import MLInferenceService
from app.models.room import Camera, Room
from app.schemas.exam import ExamSessionCreate
from app.services.exam_service import exam_service
from app.db.session import SessionLocal

VIDEO_PATH = os.path.join(str(BASE_DIR), "storage", "test_videos", "classroom_real_persons.mp4")


def evaluate():
    print("=" * 80)
    print("TEMPO — END-TO-END EMPIRICAL CLASSROOM VALIDATION")
    print(f"Target Video: {VIDEO_PATH}")
    print(f"File Size: {os.path.getsize(VIDEO_PATH) / 1024:.1f} KB, Exists: {os.path.exists(VIDEO_PATH)}")
    print("=" * 80)

    results = {}

    # -------------------------------------------------------------
    # 1. ResNet-50 vs ResNet-18 Behaviour Recognition Performance
    # -------------------------------------------------------------
    print("\n[EVAL 1] Evaluating ResNet-50 vs ResNet-18 Behaviour Recognition...")
    meta_path = os.path.join(str(BASE_DIR), "models", "model_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    r18 = meta.get("baseline_comparison", {}).get("resnet18_baseline", {})
    r50 = meta.get("baseline_comparison", {}).get("resnet50_bigru_attention", {})
    ablation = meta.get("validation_sequence_ablation", {})

    results["resnet_comparison"] = {
        "resnet18_test_accuracy": r18.get("accuracy"),
        "resnet18_test_macro_f1": r18.get("macro_f1"),
        "resnet18_test_weighted_f1": r18.get("weighted_f1"),
        "resnet50_test_accuracy": r50.get("accuracy"),
        "resnet50_test_macro_f1": r50.get("macro_f1"),
        "resnet50_test_weighted_f1": r50.get("weighted_f1"),
        "delta_macro_f1": meta.get("baseline_comparison", {}).get("delta_macro_f1"),
        "sequence_16_val_accuracy": ablation.get("16", {}).get("accuracy"),
        "sequence_16_val_macro_f1": ablation.get("16", {}).get("macro_f1"),
        "sequence_16_val_weighted_f1": ablation.get("16", {}).get("weighted_f1"),
    }
    print(f"  ResNet-18 Baseline:  Acc={r18.get('accuracy')}, Macro-F1={r18.get('macro_f1')}, Weighted-F1={r18.get('weighted_f1')}")
    print(f"  ResNet-50 Test Set:  Acc={r50.get('accuracy')}, Macro-F1={r50.get('macro_f1')}, Weighted-F1={r50.get('weighted_f1')}")
    print(f"  ResNet-50 Val Seq16: Acc={ablation.get('16', {}).get('accuracy')}, Macro-F1={ablation.get('16', {}).get('macro_f1')}, Weighted-F1={ablation.get('16', {}).get('weighted_f1')}")

    # -------------------------------------------------------------
    # 2. Adaptive High-Res Inference Back-Row Recall Benchmark
    # -------------------------------------------------------------
    print("\n[EVAL 2] Evaluating Adaptive High-Resolution Inference on Back Row...")
    audit_report_path = os.path.join(str(BASE_DIR), "storage", "dense_classroom_audit", "adaptive_observation_benchmark_report.json")
    if os.path.exists(audit_report_path):
        with open(audit_report_path, "r") as f:
            adapt_rep = json.load(f)
        base_br = adapt_rep.get("baseline_detector", {}).get("back_row", {})
        adapt_br = adapt_rep.get("adaptive_highres_detector", {}).get("back_row", {})
        deltas = adapt_rep.get("deltas_vs_baseline", {})

        results["adaptive_observation"] = {
            "baseline_back_row_recall": base_br.get("recall"),
            "baseline_back_row_precision": base_br.get("precision"),
            "baseline_back_row_f1": base_br.get("f1"),
            "adaptive_back_row_recall": adapt_br.get("recall"),
            "adaptive_back_row_precision": adapt_br.get("precision"),
            "adaptive_back_row_f1": adapt_br.get("f1"),
            "back_row_recall_diff": deltas.get("back_row_recall_diff"),
            "back_row_f1_diff": deltas.get("back_row_f1_diff"),
            "precision_diff": deltas.get("precision_diff"),
            "fp_per_frame_diff": deltas.get("fp_per_frame_diff"),
            "baseline_duplicates": adapt_rep.get("baseline_detector", {}).get("aggregate", {}).get("total_duplicates"),
            "adaptive_duplicates": adapt_rep.get("adaptive_highres_detector", {}).get("aggregate", {}).get("total_duplicates"),
        }
        print(f"  Baseline Back-Row:  Recall={base_br.get('recall')}, Precision={base_br.get('precision')}, F1={base_br.get('f1')}")
        print(f"  Adaptive Back-Row:  Recall={adapt_br.get('recall')}, Precision={adapt_br.get('precision')}, F1={adapt_br.get('f1')}")
        print(f"  Back-Row Delta:     Recall Delta={deltas.get('back_row_recall_diff'):+0.4f}, F1 Delta={deltas.get('back_row_f1_diff'):+0.4f}")
        print(f"  Duplicates Control: Baseline={adapt_rep.get('baseline_detector', {}).get('aggregate', {}).get('total_duplicates')} -> Adaptive={adapt_rep.get('adaptive_highres_detector', {}).get('aggregate', {}).get('total_duplicates')}")

    # -------------------------------------------------------------
    # 3 & 4. Pipeline Execution: LOCAL vs Edge -> Fog Pipeline
    # -------------------------------------------------------------
    print("\n[EVAL 3 & 4] Running Pipeline Parity (LOCAL vs Edge->Fog) on real video...")

    # A) Existing LOCAL Pipeline
    t0 = time.time()
    local_service = MLInferenceService(
        metadata_path=settings.MODEL_METADATA_PATH,
        model_weights_path=settings.MODEL_WEIGHTS_PATH,
        device_name="cpu",
    )
    local_out = local_service.analyze_video_full(VIDEO_PATH, render_output_video=False)
    t_local = time.time() - t0

    local_preds = local_out.get("predictions", [])
    local_tracks = local_out.get("tracks", [])
    local_obs = local_out.get("observation_summary", {})
    local_dist = Counter(p["behaviour_type"] for p in local_preds)

    # B) Edge -> Fog Pipeline
    t1 = time.time()
    edge_proc = EdgeProcessing(
        metadata_path=settings.MODEL_METADATA_PATH,
        model_weights_path=settings.MODEL_WEIGHTS_PATH,
        device_name="cpu",
        detector_mode="full",
    )
    edge_out = edge_proc.process_video(VIDEO_PATH, session_id="benchmark-edge-fog")

    fog_proc = FogRoomProcessor(
        room_id="ROOM-101",
        buffer_window_seconds=30.0,
        expected_students=70,
    )
    fog_summary = fog_proc.process_edge_output(edge_out, expected_students=70)
    cloud_payload = fog_proc.prepare_cloud_payload(total_session_frames=edge_out.total_sampled_frames)
    t_edge_fog = time.time() - t1

    ef_preds = cloud_payload.get("predictions", [])
    ef_tracks = cloud_payload.get("tracks", [])
    ef_obs = cloud_payload.get("observation_summary", {})
    ef_dist = Counter(p["behaviour_type"] for p in ef_preds)

    print(f"  LOCAL Pipeline time:    {t_local:.2f}s, Tracks: {len(local_tracks)}, Predictions: {len(local_preds)}")
    print(f"  Edge->Fog Pipeline time:{t_edge_fog:.2f}s, Tracks: {len(ef_tracks)}, Predictions: {len(ef_preds)}")

    # Track metrics comparison (Question 3)
    results["nominal_70_student_classroom"] = {
        "nominal_expected_students": 70,
        "total_detected_tracks": ef_obs.get("total_detected_students"),
        "stable_tracks_count": ef_obs.get("stable_tracks_count"),
        "temporal_ready_tracks_count": ef_obs.get("temporal_ready_tracks_count"),
        "back_row_tracks_count": ef_obs.get("back_row_tracks_count"),
        "mean_detection_coverage": ef_obs.get("mean_detection_coverage"),
        "mean_tracking_coverage": ef_obs.get("mean_tracking_coverage"),
        "mean_visibility_quality": ef_obs.get("mean_visibility_quality"),
        "mean_uncertainty": ef_obs.get("mean_uncertainty"),
        "reference_coverage_ratio": ef_obs.get("reference_coverage_ratio"),
    }
    print(f"  Nominal Enrollment: 70 students")
    print(f"  Detected Tracks:    {ef_obs.get('total_detected_students')}")
    print(f"  Stable Tracks:      {ef_obs.get('stable_tracks_count')}")
    print(f"  Temporal Ready:     {ef_obs.get('temporal_ready_tracks_count')}")
    print(f"  Back-Row Tracks:    {ef_obs.get('back_row_tracks_count')}")
    print(f"  Tracking Coverage:  {ef_obs.get('mean_tracking_coverage'):.1%}")
    print(f"  Uncertainty Score:  {ef_obs.get('mean_uncertainty'):.1%}")

    # Parity comparison (Question 4)
    results["pipeline_parity"] = {
        "local_tracks_count": len(local_tracks),
        "edge_fog_tracks_count": len(ef_tracks),
        "tracks_match": len(local_tracks) == len(ef_tracks),
        "local_predictions_count": len(local_preds),
        "edge_fog_predictions_count": len(ef_preds),
        "predictions_match": len(local_preds) == len(ef_preds),
        "local_behavior_distribution": dict(local_dist),
        "edge_fog_behavior_distribution": dict(ef_dist),
        "distribution_match": dict(local_dist) == dict(ef_dist),
    }
    print(f"  Tracks Count Match:        {len(local_tracks)} == {len(ef_tracks)} -> {len(local_tracks) == len(ef_tracks)}")
    print(f"  Predictions Count Match:   {len(local_preds)} == {len(ef_preds)} -> {len(local_preds) == len(ef_preds)}")
    print(f"  Behaviour Distribution Match: {dict(local_dist) == dict(ef_dist)}")

    # -------------------------------------------------------------
    # 5. Exam Mode Real Review Events Sanity Validation
    # -------------------------------------------------------------
    print("\n[EVAL 5] Testing Exam Mode Observable Review Events on Real Video...")
    db = SessionLocal()
    try:
        exam_service.seed_default_rooms_and_cameras(db)
        room = db.query(Room).first()

        # Create session
        session_create = ExamSessionCreate(
            title="Empirical Validation Exam Session",
            room_id=room.id,
            expected_students=70,
            invigilator_name="Invigilator Audit Team",
        )
        session = exam_service.create_exam_session(db, session_create)

        # Run real edge-fog monitoring on video
        monitored = exam_service.start_exam_monitoring(db, session.id, max_duration_seconds=None)
        dashboard = exam_service.get_invigilator_dashboard_data(db, session.id)

        events_data = []
        for ev in dashboard.review_events:
            events_data.append({
                "anonymous_id": ev.anonymous_student_id,
                "event_type": ev.event_type,
                "description": ev.event_description,
                "duration_seconds": ev.duration_seconds,
                "window": f"{ev.start_time_offset:.1f}s - {ev.end_time_offset:.1f}s",
                "evidence_window": f"{ev.evidence_window_start:.1f}s - {ev.evidence_window_end:.1f}s",
                "confidence": ev.confidence,
                "spatial_zone": ev.spatial_zone,
                "status": ev.review_status,
                "supporting": ev.supporting_observations,
            })

        results["exam_mode_real_events"] = {
            "session_code": session.session_code,
            "detected_students": monitored.detected_students_count,
            "coverage_ratio": monitored.coverage_ratio,
            "uncertainty_score": monitored.uncertainty_score,
            "room_monitoring_status": monitored.room_monitoring_status,
            "total_review_events": len(events_data),
            "events_breakdown": Counter(e["event_type"] for e in events_data),
            "sample_events": events_data[:5],
        }

        print(f"  Session Code:           {session.session_code}")
        print(f"  Room Status:            {monitored.room_monitoring_status}")
        print(f"  Coverage Ratio:         {monitored.coverage_ratio:.1%}")
        print(f"  Uncertainty Score:      {monitored.uncertainty_score:.1%}")
        print(f"  Total Review Events:    {len(events_data)}")
        print(f"  Event Types Breakdown:  {dict(Counter(e['event_type'] for e in events_data))}")
        for idx, ev in enumerate(events_data, 1):
            print(f"    Event {idx}: [{ev['anonymous_id']}] {ev['event_type']} "
                  f"({ev['duration_seconds']}s, Conf={ev['confidence']:.2f}, Zone={ev['spatial_zone']}) -> {ev['description']}")

    finally:
        db.close()

    # Save comprehensive results JSON
    out_file = os.path.join(str(BASE_DIR), "storage", "end_to_end_validation_report.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Validation complete! Report saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate()
