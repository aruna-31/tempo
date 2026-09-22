import os
import tempfile
import uuid
from datetime import date, time
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.academic import Section, Subject
from app.models.faculty import Faculty
from app.models.room import Room
from app.models.session import ClassSession
from app.models.video import Video
from app.models.analysis import AnalysisJob


def test_rooms_and_cameras_excluded_from_prototype(
    client: TestClient,
    auth_headers_admin: dict,
    auth_headers_faculty_a: dict
):
    """
    PROTOTYPE SCOPE: Rooms & Cameras are intentionally disabled. The video source for
    the prototype is manual real MP4 upload only; camera/NVR integration is out of scope.
    The routes must NOT exist in the prototype API surface.
    """
    res_create = client.post(
        "/api/v1/rooms",
        json={"room_number": "101", "building": "Block A", "floor": 1, "capacity": 60},
        headers=auth_headers_admin
    )
    assert res_create.status_code == 404  # route removed from prototype

    res_list = client.get("/api/v1/rooms", headers=auth_headers_faculty_a)
    assert res_list.status_code == 404  # route removed from prototype


def test_schedules_and_session_instantiation(
    client: TestClient,
    auth_headers_faculty_a: dict,
    faculty_a: Faculty,
    db_session: Session
):
    # Create Subject & Section
    subject = Subject(code="CS301", name="Database Systems", faculty_id=faculty_a.id)
    db_session.add(subject)
    db_session.commit()

    section = Section(subject_id=subject.id, name="Sec-1", academic_year="2025-2026", semester="ODD")
    db_session.add(section)
    db_session.commit()

    # Create recurring schedule
    res_sched = client.post(
        "/api/v1/schedules",
        json={
            "subject_id": str(subject.id),
            "section_id": str(section.id),
            "day_of_week": "MONDAY",
            "start_time": "09:00:00",
            "end_time": "09:50:00"
        },
        headers=auth_headers_faculty_a
    )
    assert res_sched.status_code == 201

    # Generate calendar sessions across a 7-day span
    res_gen = client.post(
        "/api/v1/schedules/generate-sessions",
        json={
            "start_date": "2026-09-14",
            "end_date": "2026-09-20"
        },
        headers=auth_headers_faculty_a
    )
    assert res_gen.status_code == 200
    assert res_gen.json()["generated_count"] >= 1


def test_nvr_webhook_and_auto_ingestion(
    client: TestClient,
    db_session: Session,
    faculty_a: Faculty
):
    # Setup Room, Subject, Section, Session
    room = Room(room_number="202", building="Block B", floor=2, capacity=50)
    db_session.add(room)
    db_session.commit()

    subject = Subject(code="CS401", name="Cloud Computing", faculty_id=faculty_a.id)
    db_session.add(subject)
    db_session.commit()

    section = Section(subject_id=subject.id, name="Sec-B", academic_year="2025-2026", semester="ODD")
    db_session.add(section)
    db_session.commit()

    today = date.today()
    session = ClassSession(
        faculty_id=faculty_a.id,
        section_id=section.id,
        room_id=room.id,
        room_number="202",
        title="CS401 Lecture",
        session_date=today,
        start_time=time(15, 0, 0),
        end_time=time(15, 50, 0),
        status="SCHEDULED"
    )
    db_session.add(session)
    db_session.commit()

    # Create dummy temporary video file on disk
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_file.write(b"fake_video_bytes_header_1234567890")
        tmp_path = tmp_file.name

    try:
        # Call NVR webhook
        res_webhook = client.post(
            "/api/v1/ingest/nvr-webhook",
            json={
                "event_type": "RECORDING_COMPLETED",
                "room_number": "202",
                "recording_date": str(today),
                "start_time": "15:02:00",
                "end_time": "15:50:00",
                "file_path": tmp_path
            },
            headers={"X-NVR-Token": "tempo_nvr_secure_webhook_key_2026"}
        )

        assert res_webhook.status_code == 200
        data = res_webhook.json()
        assert data["success"] is True
        assert data["session_id"] == str(session.id)
        assert data["job_id"] is not None
        assert data["match_status"] in ("EXACT_MATCH", "TOLERANCE_MATCH")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def test_job_retry_and_logs(
    client: TestClient,
    auth_headers_faculty_a: dict,
    faculty_a: Faculty,
    db_session: Session
):
    # Setup subject, section, session, video and failed job
    subject = Subject(code="CS501", name="Distributed Systems", faculty_id=faculty_a.id)
    db_session.add(subject)
    db_session.commit()

    section = Section(subject_id=subject.id, name="Sec-D", academic_year="2025-2026", semester="ODD")
    db_session.add(section)
    db_session.commit()

    session = ClassSession(
        faculty_id=faculty_a.id,
        section_id=section.id,
        title="Test Session",
        session_date=date.today(),
        start_time=time(10, 0),
        end_time=time(11, 0)
    )
    db_session.add(session)
    db_session.commit()

    video = Video(
        session_id=session.id,
        faculty_id=faculty_a.id,
        file_path="dummy.mp4",
        original_filename="dummy.mp4",
        file_size_bytes=1024,
        status="FAILED"
    )
    db_session.add(video)
    db_session.commit()

    job = AnalysisJob(
        video_id=video.id,
        status="FAILED",
        current_stage="FAILED",
        error_message="Simulated failure for retry testing",
        processing_logs=[{"timestamp": "2026-09-16T10:00:00Z", "stage": "DETECTION", "level": "ERROR", "message": "Failed"}]
    )
    db_session.add(job)
    db_session.commit()

    # Get logs endpoint
    res_logs = client.get(f"/api/v1/analysis/jobs/{job.id}/logs", headers=auth_headers_faculty_a)
    assert res_logs.status_code == 200
    assert len(res_logs.json()["logs"]) == 1

    # Retry job endpoint
    res_retry = client.post(f"/api/v1/analysis/jobs/{job.id}/retry", headers=auth_headers_faculty_a)
    assert res_retry.status_code == 200
    assert res_retry.json()["status"] in ("QUEUED", "PROCESSING")
