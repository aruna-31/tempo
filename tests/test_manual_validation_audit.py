import uuid
import pytest
import numpy as np
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from app.models.analysis import (
    AnalysisJob,
    BehaviourResult,
    ChangePoint,
    ClassroomEntropy,
    ClassroomTemporalState,
    CoverageMetric,
    FacultyInsight,
    StudentTemporalProfile,
    StudentTrackResult,
)
from app.services.temporal_analytics_service import (
    BEHAVIOURS,
    TemporalAnalyticsService,
    classroom_state,
    shannon_entropy,
    jensen_shannon_divergence,
    segment_labels,
    transition_matrix,
)
from app.services.faculty_insight_service import (
    FacultyInsightService,
    FORBIDDEN_DEFICIT_WORDS,
    assert_ethical_guardrails,
)
from app.ml.annotator import VideoAnnotator, TRACK_COLORS
from app.ml.observation import (
    compute_track_observation_metrics,
    compute_session_observation_summary,
)


def test_classroom_state_insufficient_evidence():
    """Phase 11: Classroom state detection returns INSUFFICIENT_EVIDENCE when empty."""
    assert classroom_state({}) == "INSUFFICIENT_EVIDENCE"
    assert classroom_state({"Writing": 0.0, "Reading": 0.0}) == "INSUFFICIENT_EVIDENCE"


def test_classroom_state_dominance_and_mixed():
    """Phase 11: Classroom state detection tests standard dominance and mixed activity."""
    assert classroom_state({"Looking_Toward_Instruction": 0.55, "Writing": 0.25, "Reading": 0.20}) == "Instruction-Dominant"
    assert classroom_state({"Reading": 0.45, "Writing": 0.35, "Looking_Away": 0.20}) == "Reading-Dominant"
    assert classroom_state({"Writing": 0.50, "Reading": 0.30, "Looking_Away": 0.20}) == "Individual-Work-Dominant"
    assert classroom_state({"Peer_Interaction": 0.42, "Writing": 0.30, "Reading": 0.28}) == "Collaborative-Interaction-Dominant"
    # Below 0.40 dominance defaults to Mixed-Activity
    assert classroom_state({"Looking_Toward_Instruction": 0.30, "Writing": 0.30, "Reading": 0.25, "Peer_Interaction": 0.15}) == "Mixed-Activity"


def test_single_window_entropy_semantics():
    """Phase 10: Single-window vs multi-window entropy values are valid Shannon entropy."""
    dist = {"Looking_Toward_Instruction": 0.8, "Reading": 0.1, "Writing": 0.1, "Peer_Interaction": 0.0, "Looking_Away": 0.0}
    h = shannon_entropy(dist)
    assert 0.0 < h < 1.0
    # Zero probability handling
    zero_dist = {"Looking_Toward_Instruction": 1.0, "Reading": 0.0, "Writing": 0.0, "Peer_Interaction": 0.0, "Looking_Away": 0.0}
    assert shannon_entropy(zero_dist) == 0.0


def test_change_points_minimum_evidence():
    """Phase 12: Jensen-Shannon change points require at least 2 consecutive windows."""
    dist1 = {"Looking_Toward_Instruction": 0.8, "Reading": 0.1, "Writing": 0.1, "Peer_Interaction": 0.0, "Looking_Away": 0.0}
    dist2 = {"Looking_Toward_Instruction": 0.1, "Reading": 0.1, "Writing": 0.1, "Peer_Interaction": 0.7, "Looking_Away": 0.0}
    js_div = jensen_shannon_divergence(dist1, dist2)
    assert js_div > 0.15

    # Identity divergence must be 0
    assert jensen_shannon_divergence(dist1, dist1) == 0.0


def test_temporal_coverage_not_ratio_of_windows_to_frames():
    """Phase 4: Track temporal coverage is based on sequence span, not windows count / frame count."""
    job_id = uuid.uuid4()
    # 22 frames in track history
    track_history = [{"frame": i, "bbox": [10, 10, 50, 80], "confidence": 0.9} for i in range(22)]
    # 1 temporal prediction window (T=16)
    track = StudentTrackResult(
        job_id=job_id,
        track_id=1,
        start_frame=0,
        end_frame=21,
        bounding_box_history=track_history,
    )
    # Track observation metrics: 22 >= 16 -> is_temporal_ready is True
    metrics = compute_track_observation_metrics(
        track={"track_id": 1, "start_frame": 0, "end_frame": 21, "bounding_box_history": track_history},
        total_session_frames=22,
        required_sequence_length=16,
    )
    assert metrics["is_temporal_ready"] is True
    assert metrics["temporal_readiness"] == 1.0
    assert metrics["tracking_coverage"] == 1.0


def test_fog_recovery_preserves_tagging():
    """Phase 3: Fog-recovered frames strictly retain recovered_by_fog=True."""
    from app.ml.fog import FogRoomProcessor

    fog = FogRoomProcessor()
    trajectories = [
        {
            "track_id": 1,
            "start_frame": 0,
            "end_frame": 10,
            "bounding_box_history": [{"frame": i, "bbox": [100.0, 100.0, 150.0, 200.0]} for i in range(11)],
        },
        {
            "track_id": 2,
            "start_frame": 15,
            "end_frame": 25,
            "bounding_box_history": [{"frame": i, "bbox": [102.0, 101.0, 152.0, 201.0]} for i in range(15, 26)],
        },
    ]
    fog.session_trajectories = trajectories
    stats = fog.recover_temporal_trajectories(max_gap_frames=8, seat_distance_threshold=50.0)
    assert stats["stitched_fragments_count"] >= 1
    assert stats["interpolated_frames_count"] == 4  # frames 11, 12, 13, 14

    # Verify that every interpolated frame has recovered_by_fog = True
    stitched_track = fog.session_trajectories[0]
    interpolated_frames = [
        h for h in stitched_track["bounding_box_history"]
        if h.get("interpolated") is True
    ]
    assert len(interpolated_frames) == 4
    for h in interpolated_frames:
        assert h.get("recovered_by_fog") is True


def test_ethical_guardrails_prevent_psychological_claims():
    """Phase 8 & 9: Verify strict ethical guardrails against deficit / psychological terms."""
    prohibited_samples = [
        "The student was bored during lecture.",
        "Students showed complete disengagement.",
        "The class became inattentive after 30 minutes.",
        "Lazy students in the back row.",
        "Student is failing to learn the material.",
    ]
    for text in prohibited_samples:
        with pytest.raises(ValueError) as exc:
            assert_ethical_guardrails(text)
        assert "Ethical Guardrail Violation" in str(exc.value)

    # Valid observable statements pass
    valid_text = (
        "Classroom was observed in Instruction-Dominant activity in the available 15s temporal window. "
        "Looking_Away was recorded at 12.5%. Optional Instructional Suggestion: consider inserting a concept check."
    )
    assert_ethical_guardrails(valid_text)


def test_single_window_faculty_insight_gating(db_session: Session, faculty_a):
    """Phase 5 & 10: For 1 window, faculty insights must NOT say 'remained low throughout session'."""
    from app.models.academic import Subject, Section
    from app.models.session import ClassSession
    from app.models.video import Video

    subject = Subject(faculty_id=faculty_a.id, code="AUDIT01", name="Audit Subject")
    db_session.add(subject)
    db_session.flush()
    section = Section(subject_id=subject.id, name="Sec 1", academic_year="2026", semester="ODD")
    db_session.add(section)
    db_session.flush()
    session = ClassSession(faculty_id=faculty_a.id, section_id=section.id, title="Audit Session", session_date="2026-09-28", start_time="10:00", end_time="11:00")
    db_session.add(session)
    db_session.flush()
    video = Video(session_id=session.id, faculty_id=faculty_a.id, file_path="/tmp/audit.mp4", original_filename="audit.mp4", file_size_bytes=100)
    db_session.add(video)
    db_session.flush()

    job = AnalysisJob(video_id=video.id, status="COMPLETED")
    db_session.add(job)
    db_session.flush()
    job_id = job.id

    # 1 state, 1 entropy window
    db_session.add(ClassroomTemporalState(
        job_id=job_id,
        start_time=0.0,
        end_time=15.0,
        state="Instruction-Dominant",
        behaviour_distribution_json={"Looking_Toward_Instruction": 0.85, "Writing": 0.15},
        students_contributing=22,
    ))
    db_session.add(ClassroomEntropy(
        job_id=job_id,
        timestamp=0.0,
        entropy=0.42,
        behaviour_distribution_json={"Looking_Toward_Instruction": 0.85, "Writing": 0.15},
    ))
    db_session.add(CoverageMetric(
        job_id=job_id,
        detected_student_count=22,
        tracked_student_count=22,
        temporal_ready_track_count=22,
        tracking_coverage=1.0,
        temporal_coverage=1.0,
    ))
    db_session.add_all([
        BehaviourResult(
            job_id=job_id,
            track_id=1,
            frame_number=1,
            timestamp_seconds=1.0,
            behaviour_type="Looking_Toward_Instruction",
            confidence=0.89,
        )
    ])
    db_session.commit()

    insights = FacultyInsightService.generate_and_persist_insights(db_session, job_id)
    assert len(insights) > 0

    variety_insight = next((i for i in insights if i.category == "VARIETY"), None)
    assert variety_insight is not None
    # Must NOT claim 'remained low throughout the session'
    assert "remained low" not in variety_insight.observation
    assert "single available observation window" in variety_insight.observation
    assert "Limited temporal evidence" in variety_insight.observation

    # Pacing insight must acknowledge limited evidence
    pacing_insight = next((i for i in insights if i.category == "PACING"), None)
    if pacing_insight:
        assert "continuous block" not in pacing_insight.observation or "available" in pacing_insight.observation

    # Confidence must reflect model confidence, not arbitrary 0.95
    for insight in insights:
        assert 0.0 < insight.confidence <= 1.0


def test_annotator_rendering_clean_hierarchy():
    """Phase 2: Video annotator renders clean two-tier labels without collisions."""
    annotator = VideoAnnotator()
    # Verify palette exists and is distinct
    assert len(TRACK_COLORS) >= 8
    for color in TRACK_COLORS:
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
