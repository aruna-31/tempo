"""
Unit and integration tests for Exam Mode workflow:
Floor → Room → Exam Session → Camera Source → Edge/Fog Processing → Observable Review Events → Invigilator Dashboard.

Verifies:
1. Complete independence from faculty timetable data.
2. Observable physical review events only (never automated cheating accusations).
3. Session-specific anonymous student IDs (Student 01, Student 02...).
4. Temporal persistence, timestamps, duration, confidence, and observation coverage.
5. Evidence windows and human invigilator review workflow.
6. Unified Camera Source ingestion (Demo MP4, RTSP ready).
"""
import os
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.ml.camera import (
    BaseCameraSource,
    CameraSourceFactory,
    DemoMP4CameraSource,
    RTSPStreamCameraSource,
)
from app.models.exam import ExamReviewEvent, ExamSession
from app.models.room import Camera, Room
from app.schemas.exam import ExamEventReviewRequest, ExamSessionCreate
from app.services.exam_service import exam_service


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


VIDEO_SAMPLE = "storage/test_videos/classroom_real_persons.mp4"


def test_camera_source_abstraction():
    """Verifies BaseCameraSource, DemoMP4CameraSource, and CameraSourceFactory."""
    assert os.path.exists(VIDEO_SAMPLE), f"Sample video '{VIDEO_SAMPLE}' must exist."

    # Test DemoMP4CameraSource directly
    demo_source = DemoMP4CameraSource(
        camera_id="test-cam-1",
        device_code="CAM-TEST-01",
        video_path=VIDEO_SAMPLE,
        name="Exam Hall Cam 1",
    )
    assert demo_source.source_type == "MP4_DEMO"
    assert demo_source.is_healthy() is True

    info = demo_source.get_stream_info()
    assert info["healthy"] is True
    assert info["width"] > 0
    assert info["height"] > 0
    assert info["fps"] > 0
    assert info["duration_seconds"] > 0

    # Extract sampled frames
    frames = demo_source.extract_frames(sampling_fps=2.0, max_duration_seconds=3.0)
    assert len(frames) > 0
    assert "frame_number" in frames[0]
    assert "timestamp_seconds" in frames[0]
    assert frames[0]["frame_data"].shape[0] > 0

    # Test CameraSourceFactory
    mock_cam = Camera(
        id=uuid.uuid4(),
        device_code="CAM-F1-101",
        camera_name="Floor 1 Room 101 Front",
        source_type="MP4_DEMO",
        demo_video_path=VIDEO_SAMPLE,
    )
    factory_source = CameraSourceFactory.create_from_camera(mock_cam)
    assert isinstance(factory_source, DemoMP4CameraSource)
    assert factory_source.is_healthy() is True


def test_floor_room_hierarchy_and_seeding(db_session: Session):
    """Verifies Floor → Room → Registered Camera hierarchy query and automatic seeding."""
    hierarchy = exam_service.get_floor_room_hierarchy(db_session)
    assert len(hierarchy) >= 2, "Must contain at least 2 floors (Floor 1, Floor 2)."

    floor_1 = next((f for f in hierarchy if f.floor == 1), None)
    assert floor_1 is not None
    assert len(floor_1.rooms) >= 1
    room = floor_1.rooms[0]
    assert room.room_number.startswith("Room")
    assert len(room.cameras) >= 1

    cam = room.cameras[0]
    assert cam.is_healthy is True
    assert cam.source_type == "MP4_DEMO"


def test_create_exam_session_independent_of_timetable(db_session: Session):
    """Verifies Exam Session creation without any faculty timetable dependency."""
    rooms = db_session.query(Room).filter(Room.is_active.is_(True)).all()
    assert len(rooms) > 0
    target_room = rooms[0]

    create_data = ExamSessionCreate(
        title="Midterm Assessment: Operating Systems",
        room_id=target_room.id,
        expected_students=70,
        invigilator_name="Dr. Alan Turing",
        notes="Standard closed-book exam. Floor 1.",
    )

    session = exam_service.create_exam_session(db_session, create_data)
    assert session.id is not None
    assert session.session_code.startswith("EXAM-")
    assert session.status == "SCHEDULED"
    assert session.room_monitoring_status == "IDLE"
    assert session.expected_students == 70
    assert session.camera_id is not None
    assert session.room_id == target_room.id


def test_exam_monitoring_pipeline_and_review_events(db_session: Session):
    """
    Executes edge-fog exam monitoring on the camera source and checks:
    - Strictly observable events (never automated cheating accusations).
    - Session-specific anonymous student IDs (Student 01, Student 02...).
    - Temporal persistence, timestamps, duration, and confidence.
    - Evidence window.
    """
    rooms = db_session.query(Room).filter(Room.is_active.is_(True)).all()
    target_room = rooms[0]

    # Create exam session
    create_data = ExamSessionCreate(
        title="Final Exam: Algorithm Design",
        room_id=target_room.id,
        expected_students=70,
        invigilator_name="Prof. Donald Knuth",
    )
    exam_session = exam_service.create_exam_session(db_session, create_data)

    # Run monitoring on demo video
    monitored_session = exam_service.start_exam_monitoring(
        db=db_session,
        session_id=exam_session.id,
        max_duration_seconds=6.0,  # 6s window for rapid test execution
    )

    assert monitored_session.detected_students_count > 0
    assert monitored_session.coverage_ratio > 0.0
    assert monitored_session.status == "ACTIVE"
    assert monitored_session.room_monitoring_status in ["MONITORING", "DEGRADED_OBSERVATION"]

    # Verify observable review events
    review_events = (
        db_session.query(ExamReviewEvent)
        .filter(ExamReviewEvent.exam_session_id == exam_session.id)
        .all()
    )
    assert len(review_events) > 0, "Pipeline must generate at least 1 observable review event."

    ALLOWED_OBSERVABLE_TYPES = [
        "PEER_FACING_ORIENTATION",
        "PROLONGED_UNUSUAL_ORIENTATION",
        "FREQUENT_POSTURE_CHANGE",
        "POSSIBLE_DEVICE_INTERACTION",
        "PROLONGED_ABSENCE",
        "OUT_OF_SEAT_MOVEMENT",
    ]

    for ev in review_events:
        # 1. Strictly observable type
        assert ev.event_type in ALLOWED_OBSERVABLE_TYPES
        # Ensure NEVER automated cheating accusations
        assert "cheat" not in ev.event_type.lower()
        assert "malpractice" not in ev.event_type.lower()

        # 2. Anonymous session ID only
        assert ev.anonymous_student_id.startswith("Student ")

        # 3. Temporal parameters
        assert ev.duration_seconds > 0.0
        assert ev.end_time_offset >= ev.start_time_offset
        assert 0.0 <= ev.confidence <= 1.0

        # 4. Observation quality metrics
        assert 0.0 <= ev.observation_coverage <= 1.0
        assert 0.0 <= ev.observation_uncertainty <= 1.0

        # 5. Evidence window
        assert ev.evidence_window_start is not None
        assert ev.evidence_window_end is not None
        assert ev.evidence_window_end >= ev.evidence_window_start

        # 6. Default review status
        assert ev.review_status == "PENDING_REVIEW"


def test_invigilator_human_review_workflow(db_session: Session):
    """Verifies human invigilator review actions: CONFIRMED_OBSERVATION, DISMISSED, ACTION_TAKEN."""
    # Find any pending review event
    event = (
        db_session.query(ExamReviewEvent)
        .filter(ExamReviewEvent.review_status == "PENDING_REVIEW")
        .first()
    )
    assert event is not None, "Pending review event must exist from previous test."

    # Invigilator confirms observation
    review_req = ExamEventReviewRequest(
        review_status="CONFIRMED_OBSERVATION",
        reviewed_by="Invigilator Room 101",
        review_notes="Observed student speaking with neighbor at desk 12. Issued verbal reminder.",
    )
    reviewed_event = exam_service.review_event(db_session, event.id, review_req)

    assert reviewed_event.review_status == "CONFIRMED_OBSERVATION"
    assert reviewed_event.reviewed_by == "Invigilator Room 101"
    assert reviewed_event.reviewed_at is not None
    assert "verbal reminder" in reviewed_event.review_notes

    # Test dismissing an event
    dismiss_req = ExamEventReviewRequest(
        review_status="DISMISSED",
        reviewed_by="Invigilator Room 101",
        review_notes="Student was borrowing an authorized eraser. Disregard.",
    )
    dismissed_event = exam_service.review_event(db_session, event.id, dismiss_req)
    assert dismissed_event.review_status == "DISMISSED"


def test_invigilator_dashboard_payload(db_session: Session):
    """Verifies InvigilatorDashboardData aggregation."""
    session = db_session.query(ExamSession).first()
    assert session is not None

    dashboard = exam_service.get_invigilator_dashboard_data(db_session, session.id)

    assert dashboard.session.id == session.id
    assert "is_healthy" in dashboard.camera_health
    assert "expected_students" in dashboard.monitoring_metrics
    assert "detected_students" in dashboard.monitoring_metrics
    assert "coverage_ratio" in dashboard.monitoring_metrics
    assert "summary_counts" in dashboard.__dict__
    assert dashboard.summary_counts["total"] >= 0


def test_api_endpoints_integration(client: TestClient):
    """Verifies FastAPI REST endpoints for Exam Mode."""
    # 1. GET /api/v1/exams/floors
    res = client.get("/api/v1/exams/floors")
    assert res.status_code == 200
    floors = res.json()
    assert len(floors) >= 1
    room_id = floors[0]["rooms"][0]["id"]

    # 2. POST /api/v1/exams/sessions
    payload = {
        "title": "API Test Exam Session",
        "room_id": room_id,
        "expected_students": 65,
        "invigilator_name": "API Invigilator",
    }
    create_res = client.post("/api/v1/exams/sessions", json=payload)
    assert create_res.status_code == 201
    created = create_res.json()
    session_id = created["id"]
    assert created["status"] == "SCHEDULED"

    # 3. GET /api/v1/exams/sessions
    list_res = client.get("/api/v1/exams/sessions")
    assert list_res.status_code == 200
    assert any(s["id"] == session_id for s in list_res.json())

    # 4. POST /api/v1/exams/sessions/{id}/monitor (with small duration limit)
    mon_res = client.post(f"/api/v1/exams/sessions/{session_id}/monitor?max_duration_seconds=4.0")
    assert mon_res.status_code == 200
    monitored = mon_res.json()
    assert monitored["detected_students_count"] >= 0

    # 5. GET /api/v1/exams/sessions/{id}/dashboard
    dash_res = client.get(f"/api/v1/exams/sessions/{session_id}/dashboard")
    assert dash_res.status_code == 200
    dash = dash_res.json()
    assert "camera_health" in dash
    assert "review_events" in dash

    # 6. If review events exist, test PATCH /api/v1/exams/events/{event_id}/review
    if dash["review_events"]:
        event_id = dash["review_events"][0]["id"]
        patch_res = client.patch(
            f"/api/v1/exams/events/{event_id}/review",
            json={
                "review_status": "ACTION_TAKEN",
                "reviewed_by": "Senior Proctor",
                "review_notes": "Student reseated to front row.",
            },
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["review_status"] == "ACTION_TAKEN"
