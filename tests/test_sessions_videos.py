from fastapi.testclient import TestClient


def test_session_and_video_lifecycle(
    client: TestClient, auth_headers_faculty_a, auth_headers_faculty_b
):
    # Faculty A creates subject & section
    sub_res = client.post(
        "/api/v1/subjects/",
        headers=auth_headers_faculty_a,
        json={"code": "23CS2303", "name": "Operating Systems"}
    )
    subject_id = sub_res.json()["id"]

    sec_res = client.post(
        "/api/v1/sections/",
        headers=auth_headers_faculty_a,
        json={
            "subject_id": subject_id,
            "name": "Section B",
            "academic_year": "2025-2026",
            "semester": "Semester 5"
        }
    )
    section_id = sec_res.json()["id"]

    # Faculty A schedules a class session
    sess_res = client.post(
        "/api/v1/sessions/",
        headers=auth_headers_faculty_a,
        json={
            "section_id": section_id,
            "title": "Lecture 01: Process Scheduling",
            "session_date": "2026-09-16",
            "start_time": "09:00:00",
            "end_time": "10:30:00",
            "room_number": "C-301"
        }
    )
    assert sess_res.status_code == 201
    session_id = sess_res.json()["id"]

    # Faculty B tries to access Faculty A's session -> 404
    sess_res_b = client.get(f"/api/v1/sessions/{session_id}", headers=auth_headers_faculty_b)
    assert sess_res_b.status_code == 404

    # Faculty A registers a video for the session
    vid_res = client.post(
        "/api/v1/videos/",
        headers=auth_headers_faculty_a,
        json={
            "session_id": session_id,
            "original_filename": "lecture_01_camera_front.mp4",
            "file_size_bytes": 104857600,
            "file_path": "/uploads/videos/sample.mp4",
            "duration_seconds": 3600.0
        }
    )
    assert vid_res.status_code == 201
    video_id = vid_res.json()["id"]
    assert vid_res.json()["status"] == "UPLOADED"

    # Faculty B cannot view Faculty A's video
    vid_res_b = client.get(f"/api/v1/videos/{video_id}", headers=auth_headers_faculty_b)
    assert vid_res_b.status_code == 404

    # Faculty A updates video status
    update_vid = client.patch(
        f"/api/v1/videos/{video_id}/status",
        headers=auth_headers_faculty_a,
        json={"status": "PROCESSING"}
    )
    assert update_vid.status_code == 200
    assert update_vid.json()["status"] == "PROCESSING"
