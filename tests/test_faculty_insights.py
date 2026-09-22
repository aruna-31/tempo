import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.analysis import AnalysisJob, FacultyInsight
from app.services.faculty_insight_service import (
    FacultyInsightService,
    FORBIDDEN_DEFICIT_WORDS,
    assert_ethical_guardrails,
)
from app.services.temporal_analytics_service import TemporalAnalyticsService


def test_ethical_guardrail_sanitizer():
    """Verify that forbidden deficit labels are strictly blocked by guardrails."""
    for word in FORBIDDEN_DEFICIT_WORDS:
        with pytest.raises(ValueError) as excinfo:
            assert_ethical_guardrails(f"The student was {word} during the class session.")
        assert "Ethical Guardrail Violation" in str(excinfo.value)

    # Valid, constructive text passes without exception
    assert_ethical_guardrails(
        "Looking_Away increased during the 120s-180s segment. In future classes, consider an active pause or concept check."
    )


def test_faculty_insights_generation_and_api(
    client: TestClient, db_session: Session, auth_headers_faculty_a, auth_headers_faculty_b
):
    """End-to-end test for insight generation, database persistence, and API endpoint."""
    # 1. Setup Academic hierarchy
    sub_res = client.post(
        "/api/v1/subjects/",
        headers=auth_headers_faculty_a,
        json={"code": "23CS3101", "name": "Operating Systems"},
    )
    assert sub_res.status_code == 201
    subject_id = sub_res.json()["id"]

    sec_res = client.post(
        "/api/v1/sections/",
        headers=auth_headers_faculty_a,
        json={
            "subject_id": subject_id,
            "name": "Section OS-1",
            "academic_year": "2025-2026",
            "semester": "Semester 5",
        },
    )
    assert sec_res.status_code == 201
    section_id = sec_res.json()["id"]

    st_res = client.post(
        "/api/v1/students/",
        headers=auth_headers_faculty_a,
        json={
            "section_id": section_id,
            "roll_number": "2300030555",
            "name": "Karthik R",
        },
    )
    student_id = st_res.json()["id"]

    sess_res = client.post(
        "/api/v1/sessions/",
        headers=auth_headers_faculty_a,
        json={
            "section_id": section_id,
            "title": "OS Process Scheduling",
            "session_date": "2026-09-20",
            "start_time": "09:00:00",
            "end_time": "10:30:00",
        },
    )
    session_id = sess_res.json()["id"]

    vid_res = client.post(
        "/api/v1/videos/",
        headers=auth_headers_faculty_a,
        json={
            "session_id": session_id,
            "original_filename": "os_scheduling_cam.mp4",
            "file_size_bytes": 10485760,
            "file_path": "/uploads/videos/os_cam.mp4",
            "duration_seconds": 180.0,
        },
    )
    video_id = vid_res.json()["id"]

    job_res = client.post(
        "/api/v1/analysis/jobs",
        headers=auth_headers_faculty_a,
        json={"video_id": video_id, "config_json": {}},
    )
    job_id = job_res.json()["id"]

    # 2. Add Track Results (Tracks 1 and 2)
    client.post(
        f"/api/v1/analysis/jobs/{job_id}/tracks",
        headers=auth_headers_faculty_a,
        json={
            "job_id": job_id,
            "student_id": student_id,
            "track_id": 1,
            "start_frame": 0,
            "end_frame": 3600,
            "bounding_box_history": [
                {"frame": 0, "timestamp": 0.0, "bbox": [10, 10, 100, 200]},
                {"frame": 30, "timestamp": 1.0, "bbox": [10, 10, 100, 200]},
                {"frame": 60, "timestamp": 2.0, "bbox": [10, 10, 100, 200]},
            ],
        },
    )
    client.post(
        f"/api/v1/analysis/jobs/{job_id}/tracks",
        headers=auth_headers_faculty_a,
        json={
            "job_id": job_id,
            "student_id": None,
            "track_id": 2,
            "start_frame": 0,
            "end_frame": 3600,
            "bounding_box_history": [
                {"frame": 0, "timestamp": 0.0, "bbox": [150, 10, 250, 200]},
                {"frame": 30, "timestamp": 1.0, "bbox": [150, 10, 250, 200]},
                {"frame": 60, "timestamp": 2.0, "bbox": [150, 10, 250, 200]},
            ],
        },
    )

    # 3. Add behaviour sequences simulating instruction, looking away, and peer interaction
    behaviours_data = []
    # 0s - 90s: Extended Instruction
    for t in range(0, 90, 5):
        behaviours_data.append({
            "job_id": job_id,
            "track_id": 1,
            "frame_number": t * 30,
            "timestamp_seconds": float(t),
            "behaviour_type": "Looking_Toward_Instruction",
            "confidence": 0.92,
        })
        behaviours_data.append({
            "job_id": job_id,
            "track_id": 2,
            "frame_number": t * 30,
            "timestamp_seconds": float(t),
            "behaviour_type": "Looking_Toward_Instruction",
            "confidence": 0.88,
        })

    # 90s - 130s: Gaze shifts / Looking Away
    for t in range(95, 130, 5):
        behaviours_data.append({
            "job_id": job_id,
            "track_id": 1,
            "frame_number": t * 30,
            "timestamp_seconds": float(t),
            "behaviour_type": "Looking_Away",
            "confidence": 0.85,
        })
        behaviours_data.append({
            "job_id": job_id,
            "track_id": 2,
            "frame_number": t * 30,
            "timestamp_seconds": float(t),
            "behaviour_type": "Looking_Toward_Instruction",
            "confidence": 0.80,
        })

    # 135s - 180s: Collaborative Peer Interaction
    for t in range(135, 180, 5):
        behaviours_data.append({
            "job_id": job_id,
            "track_id": 1,
            "frame_number": t * 30,
            "timestamp_seconds": float(t),
            "behaviour_type": "Peer_Interaction",
            "confidence": 0.90,
        })
        behaviours_data.append({
            "job_id": job_id,
            "track_id": 2,
            "frame_number": t * 30,
            "timestamp_seconds": float(t),
            "behaviour_type": "Peer_Interaction",
            "confidence": 0.91,
        })

    bh_res = client.post(
        f"/api/v1/analysis/jobs/{job_id}/behaviours",
        headers=auth_headers_faculty_a,
        json={"job_id": job_id, "results": behaviours_data},
    )
    assert bh_res.status_code == 201

    # 4. Generate temporal analytics
    parsed_uuid = uuid.UUID(job_id)
    TemporalAnalyticsService.build_and_persist(db_session, parsed_uuid)

    # 5. Generate and persist faculty insights
    insights = FacultyInsightService.generate_and_persist_insights(db_session, parsed_uuid)
    assert len(insights) >= 3

    categories = {i.category for i in insights}
    assert "PACING" in categories
    assert "ATTENTION_PATTERNS" in categories
    assert "INTERACTION" in categories
    assert "VARIETY" in categories
    assert "COVERAGE" in categories

    for ins in insights:
        assert ins.job_id == parsed_uuid
        assert len(ins.observation) > 10
        assert len(ins.pedagogical_context) > 10
        assert len(ins.suggested_action) > 10
        assert ins.confidence > 0.5
        # Ensure zero forbidden deficit words
        assert_ethical_guardrails(ins.observation)
        assert_ethical_guardrails(ins.pedagogical_context)
        assert_ethical_guardrails(ins.suggested_action)

    # 6. Test API endpoint GET /jobs/{job_id}/faculty-insights
    api_res = client.get(
        f"/api/v1/analysis/jobs/{job_id}/faculty-insights",
        headers=auth_headers_faculty_a,
    )
    assert api_res.status_code == 200
    res_data = api_res.json()
    assert isinstance(res_data, list)
    assert len(res_data) == len(insights)
    first = res_data[0]
    assert "category" in first
    assert "observation" in first
    assert "pedagogical_context" in first
    assert "suggested_action" in first
    assert "confidence" in first

    # Faculty B cannot view Faculty A's insights
    unauth_res = client.get(
        f"/api/v1/analysis/jobs/{job_id}/faculty-insights",
        headers=auth_headers_faculty_b,
    )
    assert unauth_res.status_code == 404


def test_faculty_insights_cascade_delete(client: TestClient, db_session: Session, auth_headers_faculty_a):
    """Verify that deleting a job cascades and removes its faculty insights."""
    # Setup minimal job
    sub_res = client.post(
        "/api/v1/subjects/",
        headers=auth_headers_faculty_a,
        json={"code": "23CS9999", "name": "Cascade Test Subject"},
    )
    subject_id = sub_res.json()["id"]

    sec_res = client.post(
        "/api/v1/sections/",
        headers=auth_headers_faculty_a,
        json={
            "subject_id": subject_id,
            "name": "Sec-Cascade",
            "academic_year": "2025-2026",
            "semester": "Semester 5",
        },
    )
    section_id = sec_res.json()["id"]

    sess_res = client.post(
        "/api/v1/sessions/",
        headers=auth_headers_faculty_a,
        json={
            "section_id": section_id,
            "title": "Cascade Session",
            "session_date": "2026-09-21",
            "start_time": "14:00:00",
            "end_time": "15:00:00",
        },
    )
    session_id = sess_res.json()["id"]

    vid_res = client.post(
        "/api/v1/videos/",
        headers=auth_headers_faculty_a,
        json={
            "session_id": session_id,
            "original_filename": "cascade.mp4",
            "file_size_bytes": 1024,
            "file_path": "/tmp/cascade.mp4",
            "duration_seconds": 60.0,
        },
    )
    video_id = vid_res.json()["id"]

    job_res = client.post(
        "/api/v1/analysis/jobs",
        headers=auth_headers_faculty_a,
        json={"video_id": video_id, "config_json": {}},
    )
    job_id = job_res.json()["id"]
    job_uuid = uuid.UUID(job_id)

    # Insert a faculty insight manually
    insight = FacultyInsight(
        job_id=job_uuid,
        category="PACING",
        observation="Test observation",
        pedagogical_context="Test context",
        suggested_action="Test action",
        confidence=0.9,
    )
    db_session.add(insight)
    db_session.commit()

    assert db_session.query(FacultyInsight).filter_by(job_id=job_uuid).count() == 1

    # Delete the job
    job = db_session.query(AnalysisJob).filter_by(id=job_uuid).first()
    db_session.delete(job)
    db_session.commit()

    # Verify cascade delete
    assert db_session.query(FacultyInsight).filter_by(job_id=job_uuid).count() == 0
