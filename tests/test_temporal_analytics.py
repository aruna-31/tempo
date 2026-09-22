from app.services.temporal_analytics_service import (
    classroom_state,
    jensen_shannon_divergence,
    segment_labels,
    shannon_entropy,
    transition_matrix,
)
from app.models.academic import Section, Subject
from app.models.analysis import AnalysisJob, BehaviourResult, StudentTrackResult
from app.models.session import ClassSession
from app.models.video import Video


def test_consecutive_labels_collapse_into_segments():
    segments = segment_labels([
        {"behaviour_type": "Writing", "timestamp_seconds": 0, "frame_number": 0, "confidence": 0.8},
        {"behaviour_type": "Writing", "timestamp_seconds": 2, "frame_number": 2, "confidence": 0.9},
        {"behaviour_type": "Reading", "timestamp_seconds": 4, "frame_number": 4, "confidence": 0.7},
    ])
    assert [item["behaviour"] for item in segments] == ["Writing", "Reading"]
    assert segments[0]["duration_seconds"] == 2


def test_transition_matrix_counts_segments_not_frames():
    counts, probabilities, transitions = transition_matrix(["Writing", "Writing", "Reading", "Reading", "Writing", "Peer_Interaction"])
    writing = 2
    reading = 1
    assert counts[writing][reading] == 1
    assert counts[reading][writing] == 1
    assert probabilities[writing][reading] == 0.5
    assert {item["count"] for item in transitions} == {1}


def test_entropy_and_js_divergence():
    uniform = {label: 0.2 for label in ("Looking_Toward_Instruction", "Reading", "Writing", "Peer_Interaction", "Looking_Away")}
    concentrated = {label: (1.0 if index == 0 else 0.0) for index, label in enumerate(uniform)}
    assert shannon_entropy(uniform) > shannon_entropy(concentrated)
    assert jensen_shannon_divergence(uniform, concentrated) > 0
    assert jensen_shannon_divergence(uniform, uniform) == 0


def test_classroom_state_requires_aggregated_dominance():
    assert classroom_state({"Writing": 0.8, "Reading": 0.2}) == "Individual-Work-Dominant"
    assert classroom_state({"Writing": 0.3, "Reading": 0.3, "Peer_Interaction": 0.4}) == "Collaborative-Interaction-Dominant"
    assert classroom_state({"Writing": 0.25, "Reading": 0.25, "Peer_Interaction": 0.25, "Looking_Away": 0.25}) == "Mixed-Activity"


def test_empty_transition_input_is_safe():
    counts, probabilities, transitions = transition_matrix([])
    assert sum(map(sum, counts)) == 0
    assert sum(map(sum, probabilities)) == 0
    assert transitions == []


def test_temporal_profile_persists_and_enforces_job_ownership(client, db_session, faculty_a, faculty_b, auth_headers_faculty_a, auth_headers_faculty_b):
    subject = Subject(faculty_id=faculty_a.id, code="TEMP01", name="Temporal Analytics")
    db_session.add(subject)
    db_session.flush()
    section = Section(subject_id=subject.id, name="A", academic_year="2026", semester="ODD")
    db_session.add(section)
    db_session.flush()
    session = ClassSession(faculty_id=faculty_a.id, section_id=section.id, title="Temporal", session_date="2026-09-22", start_time="10:00", end_time="11:00")
    db_session.add(session)
    db_session.flush()
    video = Video(session_id=session.id, faculty_id=faculty_a.id, file_path="/tmp/temporal.mp4", original_filename="temporal.mp4", file_size_bytes=1)
    db_session.add(video)
    db_session.flush()
    job = AnalysisJob(video_id=video.id, status="COMPLETED")
    db_session.add(job)
    db_session.flush()
    db_session.add(StudentTrackResult(job_id=job.id, track_id=1, bounding_box_history=[{"frame": 1}], start_frame=1, end_frame=1))
    db_session.add_all([
        BehaviourResult(job_id=job.id, track_id=1, frame_number=1, timestamp_seconds=0.0, behaviour_type="Writing", confidence=0.8),
        BehaviourResult(job_id=job.id, track_id=1, frame_number=2, timestamp_seconds=1.0, behaviour_type="Reading", confidence=0.7),
    ])
    db_session.commit()

    response = client.get(f"/api/v1/analysis/jobs/{job.id}/temporal/profile", headers=auth_headers_faculty_a)
    assert response.status_code == 200
    assert len(response.json()["students"]) == 1
    assert response.json()["transitions"][0]["count"] == 1

    denied = client.get(f"/api/v1/analysis/jobs/{job.id}/temporal/profile", headers=auth_headers_faculty_b)
    assert denied.status_code == 404
